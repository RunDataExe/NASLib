import torch
import logging
import re
from naslib.utils.shape_tracker import ShapeTracker

logger = logging.getLogger(__name__)

class ShapeAnnotator:
    """Annotates shape information on graph edges and operations."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def annotate_graph(self, graph):
        """
        Annotate the graph with shape information.
        
        Args:
            graph: The graph to annotate
        """
        # 1. Collect shape information
        shape_info = self._collect_shape_info(graph)
        
        # 2. Annotate the graph edges with shape information
        self._annotate_edges(graph, shape_info)
        
        logger.info(f"Graph annotation completed with {len(shape_info)} shape entries")
        return graph
    
    def _collect_shape_info(self, graph):
        """
        Collect shape information by running a forward pass with a dummy input.
        
        Args:
            graph: The graph to analyze
            
        Returns:
            dict: Dictionary mapping module names to shape information
        """
        graph.eval()  # Set to evaluation mode
        tracker = ShapeTracker()
        tracker.register_hooks(graph)
        
        try:
            # Create appropriate dummy input based on dataset
            if self.config.dataset == "cifar10" or self.config.dataset == "cifar100":
                dummy_input = torch.randn(self.config.search.batch_size, 3, 32, 32).to(self.device)
            elif self.config.dataset == "ImageNet16-120":
                dummy_input = torch.randn(self.config.search.batch_size, 3, 16, 16).to(self.device)
            else:
                dummy_input = torch.randn(self.config.search.batch_size, 3, 32, 32).to(self.device)  # Default

            # Run a forward pass
            graph(dummy_input)
            
            # Get shape information
            shape_info = tracker.get_shape_info()
            return shape_info
            
        except Exception as e:
            logger.error(f"Error collecting shape information: {str(e)}")
            return {}
        finally:
            # Clean up hooks
            tracker.clear_hooks()
    
    def _annotate_edges(self, graph, shape_info):
        """
        Annotate graph edges with shape information.
        
        Args:
            graph: The graph to annotate
            shape_info: Dictionary with shape information from the tracker
        """
        # First, organize shape_info by edge and primitive for easier lookup
        edge_primitives_map = {}
        for key, shapes in shape_info.items():
            edge_match = re.search(r'makrograph-edge\((\d+),(\d+)\)\.cell-edge\((\d+),(\d+)\)\.primitive-(\d+)', key)
            if edge_match:
                makro_from, makro_to, cell_from, cell_to, primitive_idx = map(int, edge_match.groups())
                edge_key = (makro_from, makro_to, cell_from, cell_to)
                primitive_key = int(primitive_idx)
                
                if edge_key not in edge_primitives_map:
                    edge_primitives_map[edge_key] = {}
                    
                if primitive_key not in edge_primitives_map[edge_key]:
                    edge_primitives_map[edge_key][primitive_key] = {}
                
                # Check if this is for an operation within the primitive
                op_match = re.search(r'\.op\.(\d+)$', key)
                if op_match:
                    op_idx = int(op_match.group(1))
                    if 'ops' not in edge_primitives_map[edge_key][primitive_key]:
                        edge_primitives_map[edge_key][primitive_key]['ops'] = {}
                    edge_primitives_map[edge_key][primitive_key]['ops'][op_idx] = shapes
                else:
                    # This is for the primitive itself
                    edge_primitives_map[edge_key][primitive_key]['self'] = shapes
        
        # Now annotate all edges in the graph
        def update_edge(edge):
            """Update function for each edge"""
            head, tail = edge.head, edge.tail
            
            # For direct edge operations (like those in the makrograph)
            if hasattr(edge.data.op, 'seq'):
                for i, layer in enumerate(edge.data.op.seq):
                    seq_key = f"makrograph-edge({head},{tail}).seq.{i}"
                    if seq_key in shape_info:
                        if not hasattr(layer, 'shapes'):
                            layer.shapes = {}
                        layer.shapes.update(shape_info[seq_key])
            
            # For operations with primitives (MixedOp)
            if hasattr(edge.data.op, 'primitives'):
                # Find the cell nodes
                cell_nodes = None
                for key in edge_primitives_map.keys():
                    makro_from, makro_to, _, _ = key
                    if makro_from == head and makro_to == tail:
                        cell_nodes = (key[2], key[3])
                        break
                
                if cell_nodes:
                    cell_from, cell_to = cell_nodes
                    edge_key = (head, tail, cell_from, cell_to)
                    
                    # Annotate each primitive
                    for i, primitive in enumerate(edge.data.op.primitives):
                        if edge_key in edge_primitives_map and i in edge_primitives_map[edge_key]:
                            primitive_info = edge_primitives_map[edge_key][i]
                            
                            # Annotate the primitive itself
                            if 'self' in primitive_info:
                                if not hasattr(primitive, 'shapes'):
                                    primitive.shapes = {}
                                primitive.shapes.update(primitive_info['self'])
                            
                            # Annotate operations within the primitive
                            if hasattr(primitive, 'op') and isinstance(primitive.op, torch.nn.Sequential):
                                if 'ops' in primitive_info:
                                    for op_idx, op_shapes in primitive_info['ops'].items():
                                        if op_idx < len(primitive.op):
                                            if not hasattr(primitive.op[op_idx], 'shapes'):
                                                primitive.op[op_idx].shapes = {}
                                            primitive.op[op_idx].shapes.update(op_shapes)
                                            
                                            # Also ensure the primitive has input/output shapes
                                            if not hasattr(primitive, 'shapes'):
                                                primitive.shapes = {}
                                            
                                            # First op's input is primitive's input
                                            if op_idx == 0 and 'input_shape' in op_shapes and 'input_shape' not in primitive.shapes:
                                                primitive.shapes['input_shape'] = op_shapes['input_shape']
                                            
                                            # Last op's output is primitive's output
                                            if op_idx == len(primitive.op) - 1 and 'output_shape' in op_shapes:
                                                primitive.shapes['output_shape'] = op_shapes['output_shape']
                                
                            # Special case for avgpool in primitive-4
                            if hasattr(primitive, 'avgpool'):
                                avg_key = f"makrograph-edge({head},{tail}).cell-edge({cell_from},{cell_to}).primitive-{i}.avgpool"
                                if avg_key in shape_info:
                                    if not hasattr(primitive.avgpool, 'shapes'):
                                        primitive.avgpool.shapes = {}
                                    primitive.avgpool.shapes.update(shape_info[avg_key])
                                    
                                    # Also ensure the primitive has input/output shapes
                                    if not hasattr(primitive, 'shapes'):
                                        primitive.shapes = {}
                                    if 'input_shape' in shape_info[avg_key]:
                                        primitive.shapes['input_shape'] = shape_info[avg_key]['input_shape']
                                    if 'output_shape' in shape_info[avg_key]:
                                        primitive.shapes['output_shape'] = shape_info[avg_key]['output_shape']
                
                # Debug output for verification
                for i, primitive in enumerate(edge.data.op.primitives):
                    if hasattr(primitive, 'shapes'):
                        if 'input_shape' in primitive.shapes and 'output_shape' in primitive.shapes:
                            print(f"Input shape: {primitive.shapes['input_shape']} - Primitive {i}")
                            print(f"Output shape: {primitive.shapes['output_shape']} - Primitive {i}")
                    elif hasattr(primitive, 'op') and isinstance(primitive.op, torch.nn.Sequential):
                        # Check each operation inside the primitive
                        for j, op in enumerate(primitive.op):
                            if hasattr(op, 'shapes'):
                                if 'input_shape' in op.shapes and 'output_shape' in op.shapes:
                                    print(f"Input shape: {op.shapes['input_shape']} - Primitive {i}, Op {j}")
                                    print(f"Output shape: {op.shapes['output_shape']} - Primitive {i}, Op {j}")
                            else:
                                print(f"Primitive {i} operation {j} has no shape information.")
                    else:
                        print(f"Primitive {i} has no shape information.")
                        
        # Update all edges in the graph
        graph.update_edges(update_edge, scope="all", private_edge_data=True)