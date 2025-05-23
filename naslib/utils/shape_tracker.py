import torch

class ShapeTracker:
    def __init__(self):
        self.shape_info = {}
        self.hooks = []

    def _hook_fn(self, name):
        def hook(module, input_tensor, output_tensor):
            if isinstance(input_tensor, tuple):
                input_shape = tuple([x.shape if isinstance(x, torch.Tensor) else None for x in input_tensor])
            else:
                input_shape = input_tensor.shape
                
            if isinstance(output_tensor, tuple):
                output_shape = tuple([x.shape if isinstance(x, torch.Tensor) else None for x in output_tensor])
            else:
                output_shape = output_tensor.shape
                
            self.shape_info[name] = {
                'input_shape': input_shape,
                'output_shape': output_shape
            }
        return hook

    def register_hooks(self, model, prefix=""):
        """Register hooks on all operations of the model"""
        for name, module in model.named_modules():
            # Skip containers like Sequential
            if len(list(module.children())) == 0:  # It's a leaf module
                full_name = f"{prefix}.{name}" if prefix else name
                hook = module.register_forward_hook(self._hook_fn(full_name))
                self.hooks.append(hook)
                
    def clear_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_shape_info(self):
        """Return collected shape information"""
        return self.shape_info