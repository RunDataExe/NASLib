import torch
import torch.nn as nn
import torch.nn.functional as F

# from naslib.predictors import ZeroCost
from naslib.predictors.zerocost_no_post_processing import (
    ZeroCost,
)  # Ensure this import matches your project structure
import logging
import math  # Ensure math is imported for isnan/isinf if used by ZeroCost or its callees
from naslib.search_spaces.core.primitives import AbstractPrimitive  # Add this import

logger = logging.getLogger(__name__)


# def adapt_spatial(x, target_h, target_w):
#     """Adapt tensor spatial dimensions to target"""

#     h, w = x.shape[-2:]

#     if h < target_h or w < target_w:
#         pad_h = max(0, target_h - h)
#         pad_w = max(0, target_w - w)
#         padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
#         x = F.pad(x, padding)

#     if h > target_h or w > target_w:
#         start_h = (h - target_h) // 2
#         start_w = (w - target_w) // 2
#         x = x[:, :, start_h : start_h + target_h, start_w : start_w + target_w]


#     return x
def adapt_spatial(x, target_h, target_w):
    """
    Adapt tensor spatial dimensions to target using interpolation.
    This is fairer than padding or cropping as it considers all spatial information.
    """
    current_h, current_w = x.shape[-2:]

    if current_h == target_h and current_w == target_w:
        return x

    # Use bilinear interpolation for resizing. It's parameter-free and considers all input pixels.
    # 'align_corners=False' is recommended for feature maps.
    return F.interpolate(
        x, size=(target_h, target_w), mode="bilinear", align_corners=False
    )


def adapt_channels_fairly(x, target_channels):
    """
    Adapt tensor channels to target dimension.
    If downsampling, selects channels evenly using linspace.
    If upsampling, places original channels evenly within a zero-initialized target tensor.
    Assumes x is [B, C, H, W].
    """
    current_channels = x.shape[1]

    if current_channels == target_channels:
        return x

    if target_channels == 0:
        return torch.zeros(
            x.shape[0], 0, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype
        )

    if current_channels == 0:
        return torch.zeros(
            x.shape[0],
            target_channels,
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        )

    if current_channels < target_channels:  # Upsampling
        output_tensor = torch.zeros(
            x.shape[0],
            target_channels,
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        )

        if current_channels > 0:
            # Generate target indices in the output tensor where input channels will be placed.
            idx_to_place_at_in_output = torch.round(
                torch.linspace(
                    0, target_channels - 1, steps=current_channels, device=x.device
                )
            ).long()
            unique_idx_to_place_at_in_output = torch.unique(idx_to_place_at_in_output)

            if len(unique_idx_to_place_at_in_output) < current_channels:
                logger.warning(
                    f"Fair upsampling: Not enough unique target slots ({len(unique_idx_to_place_at_in_output)}) for all {current_channels} input channels when target is {target_channels}. "
                    f"Placing first {len(unique_idx_to_place_at_in_output)} input channels into available unique slots."
                )
                output_tensor.index_copy_(
                    1,
                    unique_idx_to_place_at_in_output,
                    x[:, : len(unique_idx_to_place_at_in_output), :, :],
                )
            else:
                # Enough unique slots. Place all 'current_channels' from x into the first 'current_channels' unique target slots.
                output_tensor.index_copy_(
                    1, unique_idx_to_place_at_in_output[:current_channels], x
                )

        # logger.debug(f"Fairly adapting channels (upsampling - placement): input_C={current_channels}, target_C={target_channels}, output_C={output_tensor.shape[1]}")
        return output_tensor

    else:  # Downsampling: current_channels > target_channels
        # Select target_channels indices evenly spaced from current_channels (source indices from x)
        selection_indices_from_input = torch.round(
            torch.linspace(
                0, current_channels - 1, steps=target_channels, device=x.device
            )
        ).long()
        unique_selection_indices_from_input = torch.unique(selection_indices_from_input)

        x_selected = torch.index_select(
            x, dim=1, index=unique_selection_indices_from_input
        )

        # Post-process to ensure exactly target_channels output
        if x_selected.shape[1] < target_channels:
            padding_needed = target_channels - x_selected.shape[1]
            padding_shape = list(x_selected.shape)
            padding_shape[1] = padding_needed
            padding = torch.zeros(padding_shape, device=x.device, dtype=x.dtype)
            x_selected = torch.cat([x_selected, padding], dim=1)
        elif x_selected.shape[1] > target_channels:
            x_selected = x_selected[:, :target_channels, :, :]

        # logger.debug(f"Fairly adapting channels (downsampling - selection): input_C={current_channels}, target_C={target_channels}, output_C={x_selected.shape[1]}")
        return x_selected


class SyntheticMicroArchitecture(nn.Module):
    """
    Creates a synthetic neural network for ZCP evaluation by wrapping an operation
    with appropriate input/output handling and a minimal classifier.
    Uses fixed classifier dimensionality and fair channel adaptation for operation output.
    """

    def __init__(
        self,
        operation,
        operation_input_shape_chw,
        operation_output_shape_chw,
        num_classes,
        fixed_intermediate_channel_dim=256,
    ):
        super().__init__()
        self.operation = operation

        if not (
            isinstance(operation_input_shape_chw, tuple)
            and len(operation_input_shape_chw) == 3
        ):
            raise ValueError(
                f"operation_input_shape_chw must be a tuple of (C, H, W), got {operation_input_shape_chw}"
            )

        self.op_input_C, self.op_input_H, self.op_input_W = operation_input_shape_chw
        (
            self.expected_op_output_C,
            self.expected_op_output_H,
            self.expected_op_output_W,
        ) = operation_output_shape_chw
        self._num_classes = num_classes
        self.fixed_intermediate_channel_dim = fixed_intermediate_channel_dim

        self.spatial_reducer = nn.AdaptiveAvgPool2d((1, 1))

        # classifier_in_features = self.fixed_intermediate_channel_dim
        classifier_in_features = self.expected_op_output_C
        if classifier_in_features == 0:
            logger.warning(
                "Fixed intermediate channel dim is 0. Classifier will have 0 input features. Setting to 1 to avoid error."
            )
            classifier_in_features = 1

        self.classifier = nn.Linear(classifier_in_features, self._num_classes)
        logger.info(
            f"SyntheticMicroArch created with classifier: Linear(in_features={self.classifier.in_features}, out_features={self.classifier.out_features})"
        )

        with torch.no_grad():
            self.classifier.weight.fill_(0.01)
            if self.classifier.bias is not None:
                self.classifier.bias.fill_(0)

    def forward(self, x):
        # 1. Adapt input to the operation's expected input shape
        x_adapted_input_channels = adapt_channels_fairly(x, self.op_input_C)

        x_adapted_input = adapt_spatial(
            x_adapted_input_channels, self.op_input_H, self.op_input_W
        )

        # 2. Apply the operation
        if isinstance(self.operation, AbstractPrimitive):
            op_output = self.operation(x_adapted_input, None)
        else:
            op_output = self.operation(x_adapted_input)

        # 3. Spatially reduce the operation's output
        x_pooled = self.spatial_reducer(op_output)

        # 4. Adapt channels to the fixed_intermediate_channel_dim using the fair method
        # x_channels_adapted = adapt_channels_fairly(
        #     x_pooled, self.fixed_intermediate_channel_dim
        # )

        # 5. Flatten for the classifier
        x_flattened = torch.flatten(x_pooled, start_dim=1)

        # if would use fixed and intermediat adaption
        # # 4. Adapt channels to the fixed_intermediate_channel_dim using the fair method
        # x_channels_adapted = adapt_channels_fairly(
        #     x_pooled, self.fixed_intermediate_channel_dim
        # )

        # 5. Flatten for the classifier
        # x_flattened = torch.flatten(x_channels_adapted, start_dim=1)

        if x_flattened.shape[1] != self.classifier.in_features:
            logger.warning(
                f"Flattened features {x_flattened.shape[1]} do not match classifier input features {self.classifier.in_features}. Adapting dummy if needed."
            )
            if self.classifier.in_features == 1 and x_flattened.shape[1] == 0:
                x_flattened = torch.zeros(
                    x_flattened.shape[0],
                    1,
                    device=x_flattened.device,
                    dtype=x_flattened.dtype,
                )
            elif x_flattened.shape[1] == 0 and self.classifier.in_features > 0:
                x_flattened = torch.zeros(
                    x_flattened.shape[0],
                    self.classifier.in_features,
                    device=x_flattened.device,
                    dtype=x_flattened.dtype,
                )
            # Consider if a more general reshape or error is needed if other mismatches occur
            # For now, this handles the zero-channel to classifier_in_features=1 case.

        # 6. Classify
        final_logits = self.classifier(x_flattened)
        return final_logits

    def get_loss_fn(self):
        return nn.CrossEntropyLoss()

    @property
    def num_classes(self):  # Required by ZeroCost predictor's query method
        return self._num_classes


def evaluate_micro_architecture_zcp(
    operation,
    operation_input_full_shape,
    operation_output_full_shape,
    dataloader,
    zcp_method="jacov",
    dataset="cifar10",
):
    """
    Evaluate an operation with ZCP by creating a synthetic neural network.

    Args:
        operation: Single nn.Module (e.g., a primitive operation or a sequence like ConvBNReLU).
        operation_input_full_shape: Tuple, e.g. (N, C, H, W), or (torch.Size([N,C,H,W]), None), the expected input shape for the operation.
        operation_output_full_shape: Tuple, e.g. (N, C, H, W), the expected output shape from the operation.
        dataloader: DataLoader with samples for evaluation.
        zcp_method: Zero-cost proxy method to use.
        dataset: Dataset name for dimension inference (primarily for num_classes).

    Returns:
        ZCP score for the synthetic architecture.
    """
    if zcp_method.lower() in ["params"]:
        params = sum(p.numel() for p in operation.parameters())
        # intermediate_params = torch.log(
        #     torch.tensor(params, dtype=torch.float32) / 100 + 1e-6
        # )
        # final_param = torch.sigmoid(torch.tensor(params, dtype=torch.float32)).item()
        # final_param_scaled = torch.sigmoid(
        #     torch.tensor(intermediate_params, dtype=torch.float32)
        # ).item()
        # logger.info(
        #     f"Raw params: {params:.6f}, Log-scaled params: {intermediate_params:.6f}, Sigmoid mapped without log-scaling params: {final_param:.6f}, Sigmoid mapped with log-scaling params: {final_param_scaled:.6f}"
        # )
        # return final_param_scaled
        return params

    # Determine num_classes based on the dataset
    if "cifar" in dataset.lower():
        num_classes = 100 if "100" in dataset.lower() else 10
    elif "imagenet" in dataset.lower():  # Assuming ImageNet16-120 subset or similar
        num_classes = 120
    else:
        # Fallback or raise error if dataset is unknown
        logger.warning(
            f"Unknown dataset {dataset}, defaulting to 10 classes. ZCP score might be affected."
        )
        num_classes = 10

    # Parse input shape
    shape_obj_in = None
    if isinstance(operation_input_full_shape, torch.Size):
        shape_obj_in = operation_input_full_shape
    elif (
        isinstance(operation_input_full_shape, tuple)
        and len(operation_input_full_shape) > 0
        and isinstance(operation_input_full_shape[0], torch.Size)
    ):
        shape_obj_in = operation_input_full_shape[0]
    else:
        raise ValueError(
            f"Unsupported operation_input_full_shape format: {operation_input_full_shape}"
        )

    if len(shape_obj_in) == 4:  # (N, C, H, W)
        op_input_chw = tuple(shape_obj_in[1:])
    elif len(shape_obj_in) == 3:  # (C, H, W)
        op_input_chw = tuple(shape_obj_in)
    else:
        raise ValueError(
            f"Unsupported dimensions in parsed input shape: {shape_obj_in}"
        )

    # Parse output shape
    # Assuming operation_output_full_shape is torch.Size based on logs and typical usage
    if not isinstance(operation_output_full_shape, torch.Size):
        raise ValueError(
            f"Expected operation_output_full_shape to be torch.Size, got {type(operation_output_full_shape)}"
        )

    if len(operation_output_full_shape) == 4:  # (N, C, H, W)
        op_output_chw = tuple(operation_output_full_shape[1:])
    elif len(operation_output_full_shape) == 3:  # (C, H, W)
        op_output_chw = tuple(operation_output_full_shape)
    else:
        raise ValueError(
            f"Unsupported dimensions in output shape: {operation_output_full_shape}"
        )

    logger.info(
        f"Evaluating operation {type(operation).__name__} with ZCP method: {zcp_method}"
    )
    logger.info(
        f"Op Input CHW: {op_input_chw}, Op Output CHW: {op_output_chw}, Num Classes: {num_classes}"
    )

    # Configuration for SyntheticMicroArchitecture
    fixed_intermediate_dim = 256  # Example fixed dimension
    # channel_adaptation_method is now fixed to 'fair' internally in SyntheticMicroArchitecture

    try:
        synthetic_net = SyntheticMicroArchitecture(
            operation=operation,
            operation_input_shape_chw=op_input_chw,
            operation_output_shape_chw=op_output_chw,
            num_classes=num_classes,
            fixed_intermediate_channel_dim=fixed_intermediate_dim,
        )

        # Evaluate with ZCP
        zc_predictor = ZeroCost(method_type=zcp_method)
        # The ZeroCost predictor's query method handles moving the model to the device.
        score = zc_predictor.query(graph=synthetic_net, dataloader=dataloader)

        if score is None or (
            isinstance(score, float) and (math.isnan(score) or math.isinf(score))
        ):
            logger.warning(
                f"ZCP returned problematic score ({score}) for {type(operation).__name__}. Defaulting to 0.0, which sigmoid will map to 0.5."
            )
            # For sigmoid, a raw score of 0.0 results in 0.5.
            # If a true "failure" score of 0.0 post-sigmoid is desired,
            # a very negative number could be used, e.g., -float('inf'),
            # but 0.0 raw is a neutral point for sigmoid.
            # Let's return 0.5 in this case (sigmoid(0))
            return torch.sigmoid(torch.tensor(0.0)).item()

        #! synflow has negative and positive scores?

        # Some ZCP methods like 'grasp' are lower-is-better.
        # We want higher-is-better for the optimizer, so invert if necessary.
        # Check specific ZCP literature for their interpretation.
        # Assuming synflow, jacov, snip, fisher, grad_norm are higher-is-better.
        if zcp_method.lower() in ["grasp"]:
            # For methods where lower score is better, invert the relationship.
            # Avoid division by zero. Add small epsilon if score can be 0.
            # Note: Inverting and then applying sigmoid might not be ideal.
            # Consider if the "lower-is-better" logic should be handled before sigmoid,
            # or if the raw score should be negated before sigmoid for such cases.
            # For now, applying inversion as before.
            intermediate_score = (
                1.0 / (score + 1e-10) if abs(score) < 1e-9 else 1.0 / score
            )
            logger.info(
                f"Original ZCP score ({zcp_method}): {score:.6f}, Inverted score: {intermediate_score:.6f}"
            )
        # elif zcp_method.lower() in ["params"]:
        #     params = sum(p.numel() for p in operation.parameters() if p.requires_grad)
        #     intermediate_params = torch.log(
        #         torch.tensor(params, dtype=torch.float32) / 100 + 1e-6
        #     )
        #     final_param = torch.sigmoid(torch.tensor(params, dtype=torch.float32)).item()
        #     final_param_scaled = torch.sigmoid(
        #         torch.tensor(intermediate_params, dtype=torch.float32)
        #     ).item()
        #     logger.info(
        #         f"Raw params: {params:.6f}, Log-scaled params: {intermediate_params:.6f}, Sigmoid mapped without log-scaling params: {final_param:.6f}, Sigmoid mapped with log-scaling params: {final_param_scaled:.6f}"
        #     )
        #     return final_param_scaled
        else:
            # For most ZCPs (higher is better), use as is.
            intermediate_score = float(score)
            logger.info(
                f"Intermediate ZCP score ({zcp_method}): {intermediate_score:.6f}"
            )

            # # # Apply sigmoid to map the score to [0, 1]
            # # final_score_tensor = torch.sigmoid(
            # #     torch.tensor(intermediate_score, dtype=torch.float32)
            # # )
            # # final_score = final_score_tensor.item()
            # final_score_tensor = F.softplus(
            #     torch.tensor(intermediate_score, dtype=torch.float32)
            # )
            # final_score = final_score_tensor.item()
            # logger.info(
            #     f"Intermediate score: {intermediate_score:.6f}, Softplus mapped score: {final_score:.6f}"
            # )

            # return final_score
            return intermediate_score

    except Exception as e:
        logger.error(
            f"ZCP evaluation failed for operation {type(operation).__name__} with method {zcp_method}: {str(e)}",
            exc_info=True,
        )
        # Default score on critical failure, sigmoid(0.0) = 0.5
        # If a true 0.0 is desired post-sigmoid, this should return a very negative number before sigmoid.
        # For now, returning 0.5 to indicate neutral/uncertainty due to failure.
        return torch.sigmoid(torch.tensor(0.0)).item()
