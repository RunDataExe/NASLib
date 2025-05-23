import torch
import logging
import json
import os
from typing import Dict, Any

from .shape_tracker import ShapeTracker

logger = logging.getLogger(__name__)


#It looks relative good. Primitive 4 is still missing. 
# Also it seems to do [05/23 12:54:20 nl.utils.shape_annotator]: Attaching shape information to graph operations
# [05/23 12:54:20 nl.utils.shape_annotator]: Processed 6 edges in graph makrograph-edge(2,3).cell

# and thus misses makrograph-edge(1,2)

# {
#     "makrograph-edge(1,2).seq.0": {
#         "input_shape": "(torch.Size([64, 3, 32, 32]),)",
#         "output_shape": "torch.Size([64, 16, 32, 32])"
#     },
#     "makrograph-edge(1,2).seq.1": {
#         "input_shape": "(torch.Size([64, 16, 32, 32]),)",
#         "output_shape": "torch.Size([64, 16, 32, 32])"
#     },
#     "makrograph-edge(2,3).cell-edge(1,2).primitive-0": {
#         "input_shape": "(torch.Size([64, 16, 32, 32]), None)",
#         "output_shape": "torch.Size([64, 16, 32, 32])"
#     },

# The same happens for makrograph-edge(19,20)

#     "makrograph-edge(19,20).op.0": {
#         "input_shape": "(torch.Size([64, 64, 8, 8]),)",
#         "output_shape": "torch.Size([64, 64, 8, 8])"
#     },
#     "makrograph-edge(19,20).op.1": {
#         "input_shape": "(torch.Size([64, 64, 8, 8]),)",
#         "output_shape": "torch.Size([64, 64, 8, 8])"
#     },
#     "makrograph-edge(19,20).op.2": {
#         "input_shape": "(torch.Size([64, 64, 8, 8]),)",
#         "output_shape": "torch.Size([64, 64, 1, 1])"
#     },
#     "makrograph-edge(19,20).op.3": {
#         "input_shape": "(torch.Size([64, 64, 1, 1]),)",
#         "output_shape": "torch.Size([64, 64])"
#     },
#     "makrograph-edge(19,20).op.4": {
#         "input_shape": "(torch.Size([64, 64]),)",
#         "output_shape": "torch.Size([64, 10])"
#     }
# }


class ShapeAnnotator:
    """
    Annotates graph operations with tensor shape information.
    
    This class uses the ShapeTracker to collect shape information during a forward pass
    and then maps this information to the actual operation objects in the graph.
    """
    
    def __init__(self, config):
        """
        Initialize the ShapeAnnotator with configuration parameters.
        
        Args:
            config: Configuration containing dataset and batch size information
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def annotate_graph(self, graph):
        """
        Main method to annotate a graph with shape information.
        
        Args:
            graph: The graph to annotate
            
        Returns:
            The annotated graph
        """
        # Create dummy input tensor
        input_shape = self._get_input_shape()
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Collect shape information
        shape_info = self._collect_shapes(graph, dummy_input)
        
        # Save shape info to file for debugging
        self._save_shape_info(shape_info)
        
        # Map shape information to graph operations
        self._attach_shapes_to_graph(graph, shape_info)
        
        return graph
    
    def _get_input_shape(self):
        """
        Determine the shape of the input tensor based on dataset configuration.
        
        Returns:
            Tuple representing the shape of the input tensor (batch_size, channels, height, width)
        """
        batch_size = getattr(self.config.search, "batch_size", 64)
        
        if self.config.dataset in ['cifar10', 'cifar100']:
            return (batch_size, 3, 32, 32)
        elif self.config.dataset == 'ImageNet16-120':
            return (batch_size, 3, 16, 16)
        else:
            logger.warning(f"Unknown dataset {self.config.dataset}, using default CIFAR shape")
            return (batch_size, 3, 32, 32)
    
    def _collect_shapes(self, graph, dummy_input):
        """
        Collect shape information from the graph using ShapeTracker.
        
        Args:
            graph: The graph to collect shape information from
            dummy_input: Dummy input tensor for forward pass
            
        Returns:
            Dictionary mapping module names to shape information
        """
        # Ensure graph is parsed before running forward pass
        if not graph.is_parsed:
            graph.parse()
        
        # Move graph to the correct device
        graph = graph.to(self.device)
        
        # Use ShapeTracker to collect shape information
        shape_tracker = ShapeTracker()
        shape_tracker.register_hooks(graph)
        
        # Perform forward pass to collect shape information
        with torch.no_grad():
            graph.eval()  # Set to evaluation mode
            try:
                _ = graph(dummy_input)
                logger.info("Forward pass completed successfully")
            except Exception as e:
                logger.error(f"Error during forward pass: {e}")
                raise
            
        # Get collected shape information
        shape_info = shape_tracker.get_shape_info()
        
        # Remove hooks to avoid memory leaks
        shape_tracker.clear_hooks()
        
        return shape_info
    
    def _save_shape_info(self, shape_info):
        """
        Save collected shape information to a JSON file for debugging.
        
        Args:
            shape_info: Dictionary of shape information to save
        """
        # Convert tensor shapes to strings for JSON serialization
        serializable_info = {}
        for module_name, shapes in shape_info.items():
            serializable_info[module_name] = {}
            for k, v in shapes.items():
                if isinstance(v, tuple):
                    serializable_info[module_name][k] = str(v)
                else:
                    serializable_info[module_name][k] = str(v)
        
        # Use current dataset and batch size in filename
        dataset = self.config.dataset
        batch_size = self.config.search.batch_size

        # Save to file
        filename = f"nasbench201_{dataset}_{batch_size}_shape_info.json"
        with open(filename, "w") as f:
            json.dump(serializable_info, f, indent=4)
        
        logger.info(f"Shape information saved to {filename}")
    
    def _attach_shapes_to_graph(self, graph, shape_info):
        """
        Attach shape information to the operations in the graph.
        
        Args:
            graph: The graph to annotate
            shape_info: Dictionary mapping module names to shape information
        """
        logger.info("Attaching shape information to graph operations")
        
        # Process main graph
        self._process_edges(graph, "", shape_info)
        
        return graph
    
    def _process_edges(self, graph, prefix, shape_info):
        """
        Process all edges in a graph and attach shape information to operations.
        
        Args:
            graph: The graph to process
            prefix: Module name prefix for this graph
            shape_info: Dictionary mapping module names to shape information
        """
        graph_name = graph.name if not prefix else f"{prefix}.{graph.name}"
        
        # Process all edges in the graph
        edge_count = 0
        for u, v, edge_data in graph.edges.data():
            edge_prefix = f"{graph_name}-edge({u},{v})"
            
            # Recursively process operation at this edge
            if hasattr(edge_data, "op") and edge_data.op is not None:
                self._process_op(edge_data.op, edge_prefix, shape_info)
                edge_count += 1
        
        logger.info(f"Processed {edge_count} edges in graph {graph_name}")
    
    def _process_op(self, op, prefix, shape_info):
        """
        Process an operation to attach shape information.
        
        Args:
            op: The operation to process
            prefix: Module name prefix for this operation
            shape_info: Dictionary mapping module names to shape information
        """
        # Check if the operation is a Graph (subgraph)
        if isinstance(op, torch.nn.Module) and hasattr(op, "name") and hasattr(op, "edges"):
            logger.debug(f"Processing subgraph at {prefix}")
            self._process_edges(op, prefix, shape_info)
            return
        
        # Handle operations with primitives (MixedOp, GSparseMixedOp)
        if hasattr(op, "primitives"):
            logger.debug(f"Processing mixed op with {len(op.primitives)} primitives at {prefix}")
            
            # Process each primitive
            for i, primitive in enumerate(op.primitives):
                prim_prefix = f"{prefix}.primitive-{i}"
                
                # Attach shapes to the primitive itself
                if prim_prefix in shape_info:
                    if not hasattr(primitive, 'shapes'):
                        primitive.shapes = {}
                    primitive.shapes.update(shape_info[prim_prefix])
                    logger.debug(f"Attached shape information to {prim_prefix}")
                
                # Process submodules within the primitive
                self._process_primitive(primitive, prim_prefix, shape_info)
                
        elif prefix in shape_info:
            # Direct shape information for this op
            if not hasattr(op, 'shapes'):
                op.shapes = {}
            op.shapes.update(shape_info[prefix])
            logger.debug(f"Attached shape information to {prefix}")
    
    def _process_primitive(self, primitive, prefix, shape_info):
        """
        Process a primitive and its submodules to attach shape information.
        
        Args:
            primitive: The primitive to process
            prefix: Module name prefix for this primitive
            shape_info: Dictionary mapping module names to shape information
        """
        # Handle case where primitive has an op attribute (common in NASLib operations)
        if hasattr(primitive, "op") and isinstance(primitive.op, torch.nn.Module):
            op_prefix = f"{prefix}.op"
            
            # Handle Sequential operations
            if isinstance(primitive.op, torch.nn.Sequential):
                for j, layer in enumerate(primitive.op):
                    layer_prefix = f"{op_prefix}.{j}"
                    
                    # Attach shapes to individual layers
                    if layer_prefix in shape_info:
                        if not hasattr(layer, 'shapes'):
                            layer.shapes = {}
                        layer.shapes.update(shape_info[layer_prefix])
                        logger.debug(f"Attached shape information to {layer_prefix}")
            
            # Handle single operations
            elif op_prefix in shape_info:
                if not hasattr(primitive.op, 'shapes'):
                    primitive.op.shapes = {}
                primitive.op.shapes.update(shape_info[op_prefix])
                logger.debug(f"Attached shape information to {op_prefix}")
        
        # Process named children (like conv_a, conv_b in ResNetBasicblock)
        for name, module in primitive.named_children():
            if name == "op":  # Already processed above
                continue
                
            module_prefix = f"{prefix}.{name}"
            
            # Attach shapes to this module
            if module_prefix in shape_info:
                if not hasattr(module, 'shapes'):
                    module.shapes = {}
                module.shapes.update(shape_info[module_prefix])
                logger.debug(f"Attached shape information to {module_prefix}")
            
            # If module is Sequential, process its layers too
            if isinstance(module, torch.nn.Sequential):
                for j, layer in enumerate(module):
                    layer_prefix = f"{module_prefix}.{j}"
                    
                    if layer_prefix in shape_info:
                        if not hasattr(layer, 'shapes'):
                            layer.shapes = {}
                        layer.shapes.update(shape_info[layer_prefix])
                        logger.debug(f"Attached shape information to {layer_prefix}")