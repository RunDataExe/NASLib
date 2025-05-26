import torch
import logging
import json
import os
from typing import Dict, Any
import torch.nn as nn

from .shape_tracker import ShapeTracker
from ..search_spaces.core import primitives as ops # For isinstance checks
from ..search_spaces.core.graph import Graph as CoreGraph # Import CoreGraph

logging.basicConfig(level=logging.DEBUG) # Or configure specific loggers

logger = logging.getLogger(__name__)

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
        
        # Log summary of collected shape info
        logger.info(f"Collected shape information for {len(shape_info)} modules")
        
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
        logger.debug(f"Sample keys from shape_info: {list(shape_info.keys())[:20]}") # ADD THIS LINE
        
        processed_keys = set()
        
        # Process the main graph (makrograph)
        # The initial prefix for modules directly under the main graph (e.g., ops on its edges)
        # will be constructed inside _process_main_graph_edges.
        self._process_main_graph_edges(graph, graph.name, shape_info, processed_keys)
        
        unprocessed_keys = set(shape_info.keys()) - processed_keys
        if unprocessed_keys:
            logger.warning(f"Found {len(unprocessed_keys)} unprocessed shape entries after annotation.")
            if unprocessed_keys: # Log a few examples if any exist
                logger.debug(f"Examples of unprocessed keys: {sorted(list(unprocessed_keys))[:10]}...")
        else:
            logger.info("All shape entries processed and attached to the graph.")
        
        return graph

    def _process_module_recursive(self, module_obj, current_module_key, shape_info, processed_keys):
        """
        Recursively process a module and its children to attach shape information.

        Args:
            module_obj: The nn.Module instance.
            current_module_key: The key that ShapeTracker would use for this module_obj in shape_info.
            shape_info: Dictionary mapping module names to shape information.
            processed_keys: Set to track processed keys.
        """
        is_leaf_node = not list(module_obj.children())
        is_in_shape_info = current_module_key in shape_info
        
        logger.debug(f"Processing module with key: '{current_module_key}'. Is leaf: {is_leaf_node}. Present in shape_info: {is_in_shape_info}")

        if is_leaf_node and is_in_shape_info:
            if not hasattr(module_obj, 'shapes'):
                module_obj.shapes = {}
            module_obj.shapes.update(shape_info[current_module_key])
            processed_keys.add(current_module_key)
            logger.debug(f"SUCCESS: Attached shape information to leaf module: {current_module_key} with shapes {shape_info[current_module_key]}")
            return 
        elif is_leaf_node and not is_in_shape_info:
            logger.warning(f"LEAF MODULE MISMATCH: Leaf module key '{current_module_key}' not found in shape_info keys.")

        for child_name, child_sub_module in module_obj.named_children():
            child_module_key = f"{current_module_key}.{child_name}"
            self._process_module_recursive(child_sub_module, child_module_key, shape_info, processed_keys)

    def _process_main_graph_edges(self, graph, graph_base_name, shape_info, processed_keys):
        """
        Process the edges of the main graph.
        Args:
            graph: The main graph (e.g., NasBench201SearchSpace instance).
            graph_base_name: The base name for this graph (e.g., "makrograph").
            shape_info: Dictionary mapping module names to shape information.
            processed_keys: Set to track processed keys.
        """
        logger.info(f"Processing main graph: {graph.name} (base name: {graph_base_name})")
        processed_cells_count = 0
        processed_fixed_modules_count = 0

        for u, v, edge_data in graph.edges.data():
            edge_op_base_key = f"{graph_base_name}-edge({u},{v})" # Key for the op on this edge or prefix for its children

            if hasattr(edge_data, "op") and edge_data.op is not None:
                op_module_instance = edge_data.op

                if hasattr(op_module_instance, "edges") and hasattr(op_module_instance, "name") and \
                   isinstance(op_module_instance, CoreGraph) and not isinstance(op_module_instance, ops.MixedOp):
                    logger.info(f"Processing Cell (op on edge {u},{v}), using base key for its contents: '{edge_op_base_key}'")
                    self._process_cell_edges(op_module_instance, edge_op_base_key, shape_info, processed_keys)
                    processed_cells_count += 1
                
                elif isinstance(op_module_instance, torch.nn.Module):
                    logger.info(f"Processing Fixed Module (op on edge {u},{v}), using base key for its contents: '{edge_op_base_key}'")
                    # The op_module_instance itself might not be a "leaf" in shape_info if it has children.
                    # _process_module_recursive will handle finding shape_info for its children using edge_op_base_key as prefix.
                    # If op_module_instance itself was hooked (e.g. a simple nn.Conv2d not in a Sequential), 
                    # then edge_op_base_key should be its key.
                    # However, PyTorch names from add_module usually include ".op" for the op on edge.
                    # The shape_info.json keys like "makrograph-edge(1,2).seq.0" suggest that ShapeTracker
                    # might not use the ".op" part from the PyTorch module name of the edge's op itself
                    # when forming the prefix for the children of that op.
                    # So, edge_op_base_key is the prefix for children.
                    # If the op_module_instance itself (e.g. a custom ResNetBlock) was hooked directly, its key would be edge_op_base_key.
                    # This seems to be the most consistent interpretation of the shape_info.json keys.
                    pytorch_module_name_for_op_on_edge = f"{edge_op_base_key}.op" # This is the actual PyTorch module name
                    
                    # We need to decide if shape_info keys for children of op_module_instance start with
                    # edge_op_base_key or pytorch_module_name_for_op_on_edge.
                    # Given "makrograph-edge(1,2).seq.0", it implies children of Stem (on edge 1,2)
                    # are named relative to "makrograph-edge(1,2)", not "makrograph-edge(1,2).op".
                    self._process_module_recursive(op_module_instance, edge_op_base_key, shape_info, processed_keys)
                    processed_fixed_modules_count +=1
        
        logger.info(f"Processed {processed_cells_count} cell(s) and {processed_fixed_modules_count} fixed module(s) on the main graph '{graph.name}'.")


    def _process_cell_edges(self, cell_graph, cell_graph_base_key, shape_info, processed_keys):
        """
        Process all edges within a cell graph.
        Args:
            cell_graph: The cell Graph instance.
            cell_graph_base_key: The base key for this cell graph's contents (e.g., "makrograph-edge(2,3)").
            shape_info: Dictionary mapping module names to shape information.
            processed_keys: Set to track processed keys.
        """
        edge_count = 0
        for u, v, edge_data in cell_graph.edges.data():
            cell_edge_op_base_key = f"{cell_graph_base_key}.cell-edge({u},{v})" # Key for op on this cell edge or prefix for its children
            
            if hasattr(edge_data, "op") and edge_data.op is not None:
                op_on_cell_edge_instance = edge_data.op

                if isinstance(op_on_cell_edge_instance, ops.MixedOp):
                    logger.debug(f"Processing MixedOp in cell '{cell_graph.name}' on edge ({u},{v}), using base key for its primitives: '{cell_edge_op_base_key}'")
                    self._process_mixed_op(op_on_cell_edge_instance, cell_edge_op_base_key, shape_info, processed_keys)
                elif isinstance(op_on_cell_edge_instance, torch.nn.Module):
                    # This handles cases where a specific primitive is already chosen.
                    # Similar to fixed modules on main graph, use cell_edge_op_base_key as prefix for its children.
                    logger.debug(f"Processing direct nn.Module in cell '{cell_graph.name}' on edge ({u},{v}), using base key for its contents: '{cell_edge_op_base_key}'")
                    self._process_module_recursive(op_on_cell_edge_instance, cell_edge_op_base_key, shape_info, processed_keys)
                edge_count += 1
        
        logger.info(f"Processed {edge_count} edges in cell graph '{cell_graph.name}' (base key '{cell_graph_base_key}')")

    def _process_mixed_op(self, mixed_op_instance, mixed_op_base_key, shape_info, processed_keys):
        """
        Process a MixedOp and its primitives.
        Args:
            mixed_op_instance: The MixedOp nn.Module instance.
            mixed_op_base_key: The base key for this MixedOp's primitives (e.g., "makrograph-edge(2,3).cell-edge(1,2)").
            shape_info: Dictionary mapping module names to shape information.
            processed_keys: Set to track processed keys.
        """
        if not hasattr(mixed_op_instance, "primitives") or not isinstance(mixed_op_instance.primitives, list):
            logger.warning(f"MixedOp with base key {mixed_op_base_key} has no 'primitives' list or it's not a list.")
            return
            
        for i, primitive_module_instance in enumerate(mixed_op_instance.primitives):
            primitive_base_key = f"{mixed_op_base_key}.primitive-{i}" # Key for this primitive or prefix for its children
            
            if not isinstance(primitive_module_instance, torch.nn.Module):
                logger.warning(f"Primitive at index {i} with base key {primitive_base_key} is not an nn.Module. Skipping.")
                continue

            # Initialize shapes on the primitive itself, ensuring the attribute exists.
            if not hasattr(primitive_module_instance, 'shapes'):
                primitive_module_instance.shapes = {}

            # Recursively process the primitive and its children.
            # This will attach shapes to any nn.Module components, including potentially
            # primitive_module_instance itself if primitive_base_key is in shape_info,
            # or its named children (e.g., primitive_module_instance.avgpool).
            self._process_module_recursive(primitive_module_instance, primitive_base_key, shape_info, processed_keys)

            # After recursion, check if the primitive_module_instance itself has its shapes.
            # If not, attempt to infer them from its constituent child modules.
            has_input_shape = 'input_shape' in primitive_module_instance.shapes
            has_output_shape = 'output_shape' in primitive_module_instance.shapes

            if not (has_input_shape and has_output_shape):
                logger.debug(f"Primitive {primitive_base_key} ({type(primitive_module_instance).__name__}) may lack direct shapes. Attempting to infer from children.")

                # Specific handling for ops.AvgPool1x1
                if isinstance(primitive_module_instance, ops.AvgPool1x1):
                    # Determine the child providing input and the child providing final output
                    input_child = primitive_module_instance.avgpool
                    output_child = primitive_module_instance.bn if hasattr(primitive_module_instance, 'bn') and primitive_module_instance.bn is not None else primitive_module_instance.avgpool
                    
                    if not has_input_shape and hasattr(input_child, 'shapes') and 'input_shape' in input_child.shapes:
                        primitive_module_instance.shapes['input_shape'] = input_child.shapes['input_shape']
                        logger.debug(f"Inferred input_shape for {primitive_base_key} from child {type(input_child).__name__}.")
                    
                    if not has_output_shape and hasattr(output_child, 'shapes') and 'output_shape' in output_child.shapes:
                        primitive_module_instance.shapes['output_shape'] = output_child.shapes['output_shape']
                        logger.debug(f"Inferred output_shape for {primitive_base_key} from child {type(output_child).__name__}.")

                # Specific handling for ops.ReLUConvBN (as a fallback, though it usually gets direct shapes)
                elif isinstance(primitive_module_instance, ops.ReLUConvBN):
                    if hasattr(primitive_module_instance, 'op') and isinstance(primitive_module_instance.op, nn.Sequential) and len(primitive_module_instance.op) > 0:
                        first_component = primitive_module_instance.op[0]
                        last_component = primitive_module_instance.op[-1]

                        if not has_input_shape and hasattr(first_component, 'shapes') and 'input_shape' in first_component.shapes:
                             primitive_module_instance.shapes['input_shape'] = first_component.shapes['input_shape']
                             logger.debug(f"Inferred input_shape for {primitive_base_key} from its first op component.")
                        if not has_output_shape and hasattr(last_component, 'shapes') and 'output_shape' in last_component.shapes:
                             primitive_module_instance.shapes['output_shape'] = last_component.shapes['output_shape']
                             logger.debug(f"Inferred output_shape for {primitive_base_key} from its last op component.")
                
                elif isinstance(primitive_module_instance, (ops.Identity, ops.Zero)):
                    # For Identity and Zero, if they don't have direct shapes, it's harder to infer
                    # without knowing the input shape from the graph flow.
                    # The optimizer might need to handle this by assuming output_shape = input_shape.
                    if not (has_input_shape and has_output_shape):
                        logger.debug(f"Primitive {primitive_base_key} ({type(primitive_module_instance).__name__}) is Identity/Zero and still missing direct shapes. Input shape might need to be propagated by optimizer if not found.")


            # Final check for logging, to be seen by the optimizer's context
            if not primitive_module_instance.shapes.get('output_shape') or not primitive_module_instance.shapes.get('input_shape'):
                # This warning is important for the optimizer if shapes are still missing.
                pass # The optimizer itself will log "Primitive X has no shape information."
            else:
                logger.debug(f"Shapes for {primitive_base_key} ({type(primitive_module_instance).__name__}): IN={primitive_module_instance.shapes.get('input_shape')}, OUT={primitive_module_instance.shapes.get('output_shape')}")