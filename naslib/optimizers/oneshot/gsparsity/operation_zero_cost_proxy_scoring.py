import torch
import torch.nn as nn
import torch.nn.functional as F
from naslib.predictors import ZeroCost
import logging
from typing import Union, Tuple # Add this import
from naslib.search_spaces.core.primitives import AbstractPrimitive, Identity, Zero, ReLUConvBN, AvgPool1x1
from naslib.search_spaces.nasbench201.primitives import ResNetBasicblock
logger = logging.getLogger(__name__)


#! Start over again. Only procced when task at hand is correctly implemented

#! 1) infer the input dimension of the first operation
#! 2) infer the output dimension of the last operation
#! 3) Adapt input Data to the dimension of the input operation
#! 4) Adapt output dimension of the last operation to the input dimensionality of the fixed, minimaly gradient influencing, classifier
#! 5) Apply ZCP to the new synthetic architecture 


def adapt_channels(x, target_channels):
    """Adapt tensor channels to target dimension"""
    
    logger.info(f"Adapting channels: input_shape={x.shape}, target_channels={target_channels}")
    
    current_channels = x.shape[1]
    
    if current_channels < target_channels:
        padding = torch.zeros(x.shape[0], target_channels - current_channels, 
                             *x.shape[2:], device=x.device)
        x = torch.cat([x, padding], dim=1)
    elif current_channels > target_channels:
        x = x[:, :target_channels]
    
    # After adaptation
    logger.info(f"After adaptation: output_shape={x.shape}")
    
    return x

def adapt_spatial(x, target_h, target_w):
    """Adapt tensor spatial dimensions to target"""
        
    h, w = x.shape[-2:]
    
    if h < target_h or w < target_w:
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        padding = (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2)
        x = F.pad(x, padding)
    
    if h > target_h or w > target_w:
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        x = x[:, :, start_h:start_h+target_h, start_w:start_w+target_w]
    
    return x





# TODO find out how stage can be infered or if I should use try/except

# TODO go through each primitive and verify it
# TODO Conv2d and verify if it





def get_primitive_in_out_channels(primitive: AbstractPrimitive, stage_C: int = None):
    """
    Determines the input and output channel dimensionality of a given NASLib primitive.

    Args:
        primitive (AbstractPrimitive): The NASLib primitive instance.
        stage_C (int, optional): The number of channels for the current stage/cell.
            Required for primitives like Identity, Zero(stride=1), and AvgPool1x1(stride=1)
            as used in NASBench-201 cells, where channel count is implicit from the stage.

    Returns:
        tuple(int | None, int | None): A tuple (in_channels, out_channels).
                                       Returns (None, None) if dimensionality cannot be determined.
    """
    
    import pudb
    pudb.set_trace()
    if not isinstance(primitive, AbstractPrimitive):
        raise TypeError(f"Expected an AbstractPrimitive, got {type(primitive)}")
    

    # check if each has it 
    init_params = primitive.init_params



    if isinstance(primitive, Identity):
        # Identity preserves channels. For NASBench-201 cell ops, it relies on stage_C.
        if stage_C is not None:
            return stage_C, stage_C
        else:
            # If C_in/C_out are somehow in init_params for a generic Identity (not typical for NB201 cell ops)
            c_in = init_params.get('C_in')
            c_out = init_params.get('C_out')
            if c_in is not None and c_out is not None:
                return c_in, c_out
            print(f"Warning: stage_C not provided for Identity primitive {primitive}. Cannot determine channels directly from init_params {init_params}.")
            return None, None

    elif isinstance(primitive, Zero):
        stride = init_params.get('stride')
        c_in = init_params.get('C_in')
        c_out = init_params.get('C_out')

        if stride == 1:
            # Zero(stride=1) in NASBench-201 cells preserves channels, relying on stage_C.
            if c_in is not None and c_out is not None: # If explicitly defined
                 return c_in, c_out
            if stage_C is not None:
                return stage_C, stage_C
            else:
                print(f"Warning: stage_C not provided for Zero(stride=1) primitive {primitive} and C_in/C_out not in init_params {init_params}. Cannot determine channels.")
                return None, None
        else:
            # For Zero with stride > 1, C_in and C_out might be specified
            # (e.g., if it's also handling channel changes per NASLib's Zero op definition)
            if c_in is not None and c_out is not None:
                return c_in, c_out
            # If C_out is missing, it's ambiguous for a channel-changing Zero op without more rules.
            # NASLib's Zero op can infer C_out based on C_in and stride if C_in == C_out is not met,
            # but that logic is in its forward pass, not trivially in init_params alone if C_out is absent.
            print(f"Warning: Zero(stride={stride}) with init_params {init_params} and no stage_C. Channel determination might be incomplete.")
            return c_in, c_out # Returns what's available

    elif isinstance(primitive, AvgPool1x1):
        # For NASBench-201 cell ops, AvgPool1x1 has stride=1 and no C_in/C_out in init_params.
        stride = init_params.get('stride')
        if stride == 1:
            if stage_C is not None:
                return stage_C, stage_C
            else:
                # Check if C_in/C_out are in init_params (not for NB201 cell version)
                c_in = init_params.get('C_in')
                c_out = init_params.get('C_out')
                if c_in is not None and c_out is not None:
                    return c_in, c_out
                print(f"Warning: stage_C not provided for AvgPool1x1(stride=1) primitive {primitive}. Cannot determine channels from init_params {init_params}.")
                return None, None
        else:
            # AvgPool1x1 with stride > 1 should have C_in, C_out in init_params
            # as per naslib/search_spaces/core/primitives.py
            c_in = init_params.get('C_in')
            c_out = init_params.get('C_out')
            if c_in is not None and c_out is not None:
                return c_in, c_out
            else:
                print(f"Warning: AvgPool1x1(stride={stride}) missing C_in/C_out in init_params {init_params}. Cannot determine channels.")
                return None, None
    
    elif isinstance(primitive, ReLUConvBN):
        # is this even called?
        # if yes is the conv in it changing the channels?
        return init_params.get('C_in'), init_params.get('C_out')

    elif isinstance(primitive, ResNetBasicblock):
        # is this even called?
        # if yes is the conv in it changing the channels?
        return init_params.get('C_in'), init_params.get('C_out')

    # # Fallback for other primitives: try to find C_in, C_out, or C
    # c_in_generic = init_params.get('C_in')
    # c_out_generic = init_params.get('C_out')
    # if c_in_generic is not None and c_out_generic is not None:
    #     return c_in_generic, c_out_generic

    # # If only 'C' is present (e.g. some primitives might use C_in=C, C_out=C)
    # c_channel = init_params.get('C')
    # if c_channel is not None and c_in_generic is None and c_out_generic is None:
    #     return c_channel, c_channel
    
    # if c_in_generic is not None and c_out_generic is None and stage_C is not None and c_in_generic == stage_C:
    #     # If C_in matches stage_C and C_out is missing, assume C_out is also stage_C for channel-preserving ops
    #     print(f"Warning: Primitive {type(primitive).__name__} has C_in={c_in_generic} matching stage_C={stage_C} but C_out is missing. Assuming C_out={stage_C}.")
    #     return stage_C, stage_C


    print(f"Warning: Dimensionality for primitive type {type(primitive).__name__} with params {init_params} could not be determined with the current logic. stage_C was {stage_C}.")
    return None, None

class SyntheticMicroArchitecture(nn.Module):
    """
    Creates a synthetic neural network for ZCP evaluation by wrapping operation
    with appropriate input/output handling and a minimal classifier.
    """
    def __init__(self, operation, data_input_channels=3, data_input_size=(32, 32), num_classes=10):
        super().__init__()
        self.operation = operation
        self.data_input_channels = data_input_channels
        self.data_input_size = data_input_size
        self.num_classes = num_classes

        self.operation_input_dimensions, self.operation_output_dimensions = get_primitive_in_out_channels(primitive=self.operation, )
        # self.operation_output_dimensions = get_operation_output_dimensions(self.operation)
        
        # Simple classifier head
        self.classifier = nn.Linear(self.operation_output_dimensions[0], num_classes)
        with torch.no_grad():
            # Initialize with small weights to minimize impact on ZCP metrics
            self.classifier.weight.fill_(0.01)
            self.classifier.bias.fill_(0)
        

    def forward(self, x):
        x = adapt_channels(x, self.operation_input_dimensions)
        x = adapt_spatial(x, *self.operation_input_dimensions)

        for op in self.operation:
            x = op(x)

        x = adapt_channels(x, self.operation_output_dimensions)
        x = adapt_spatial(x, *self.operation_output_dimensions)

        x = self.classifier(x)

def evaluate_micro_architecture_zcp(operation, dataloader, zc_method='jacov', dataset='cifar10'):
    """
    Evaluate operation with ZCP by creating a synthetic neural network
    
    Args:
        operation: Single operation or list of operation to evaluate
        dataloader: DataLoader with samples for evaluation
        zc_method: Zero-cost proxy method to use
        dataset: Dataset name for dimension inference
        
    Returns:
        ZCP score for the synthetic architecture
    """
    # Set input data dimensions based on dataset
    if 'cifar' in dataset.lower():
        data_input_size = (32, 32)
        data_input_channels = 3
        num_classes = 100 if '100' in dataset.lower() else 10
    elif 'imagenet' in dataset.lower():
        data_input_size = (16, 16)
        data_input_channels = 3
        num_classes = 120
    
    try:
        synthetic_net = SyntheticMicroArchitecture(
            operation=operation,
            data_input_channels=data_input_channels,
            data_input_size=data_input_size,
            num_classes=num_classes
        )
        
        # Evaluate with ZCP
        zc_predictor = ZeroCost(method_type=zc_method)
        score = zc_predictor.query(graph=synthetic_net, dataloader=dataloader)
        
        if score is None:
            logger.warning("ZCP returned None score")
            return 0.0  # Default score
            
        if zc_method.lower() in ['grasp']:
            # For methods where lower score is better, invert the relationship
            final_score = 1.0 / (score if score != 0 else 1e-10)
            logger.info(f"ZCP score: {score:.6f}, inverted score: {final_score:.6f}")
        else:
            # For most ZCPs (higher is better), use as is
            final_score = score
            logger.info(f"ZCP score: {final_score:.6f}")
        
        return final_score
        
    except Exception as e:
        logger.warning(f"ZCP evaluation failed: {str(e)}")
        return 0.0  # Default score
