import logging
import torch
import numpy as np
from copy import deepcopy
from pathlib import Path
import json

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.oneshot.gsparsity.zcp_gsparse_optimizer import ZCP_GSparseOptimizer
from naslib.optimizers.oneshot.gsparsity.inverted_bananas_optimizer import Inverted_Bananas
from naslib.utils import get_zc_benchmark_api
from naslib.search_spaces.core.query_metrics import Metric

logger = logging.getLogger(__name__)

class Inverted_Bananas_ZCP_GsparseOptimizer(MetaOptimizer):
    """
    A two-stage optimizer that first removes poor architectures using Inverted BANANAS,
    then applies ZCP-enhanced GSparsity to the reduced search space.
    
    Stage 1: Identify and remove worst architectures
    Stage 2: Apply ZCP-enhanced one-shot training on the reduced space
    """

    def __init__(self, config):
        """
        Initialize the two-stage optimizer with strict config adherence
        
        Args:
            config: Configuration with settings for both stages
        """
        super(Inverted_Bananas_ZCP_GsparseOptimizer, self).__init__()
        self.config = config
        self.dataset = config.dataset
        
        # Extract parameters from config with no fallbacks
        if not hasattr(config.stage1, 'epochs'):
            raise ValueError("Missing required configuration: config.stage1.epochs")
        if not hasattr(config.stage2, 'epochs'):
            raise ValueError("Missing required configuration: config.stage2.epochs")
        
        # Use exact values from config
        self.removal_percentage = config.stage1.removal_percentage
        self.stage1_epochs = config.stage1.epochs
        self.stage2_epochs = config.stage2.epochs
        
        # Extract ZCP parameters - strict mode, no fallbacks
        if not hasattr(config.stage2, 'zcp_method') and not hasattr(config.search, 'zcp_method'):
            raise ValueError("Missing required configuration: zcp_method in either stage2 or search config")
        if not hasattr(config.stage2, 'zcp_weight') and not hasattr(config.search, 'zcp_weight'):
            raise ValueError("Missing required configuration: zcp_weight in either stage2 or search config")
        
        # Use config values directly with minimal fallback between stage2 and search
        self.zcp_method = config.stage2.zcp_method if hasattr(config.stage2, 'zcp_method') else config.search.zcp_method
        self.zcp_weight = config.stage2.zcp_weight if hasattr(config.stage2, 'zcp_weight') else config.search.zcp_weight
        
        # Ensure scheduler attributes are properly set for trainer
        required_scheduler_attrs = ['learning_rate', 'learning_rate_min', 'momentum', 'weight_decay']
        for attr in required_scheduler_attrs:
            if not hasattr(config.search, attr) and hasattr(config.stage2, attr):
                setattr(config.search, attr, getattr(config.stage2, attr))
        
        # Create configs for each stage without defaults
        from fvcore.common.config import CfgNode
        
        # Create direct copies of the stage-specific configs
        stage1_config = CfgNode({
            "dataset": config.dataset,
            "search_space": config.search_space if hasattr(config, 'search_space') else None,
            "data": config.data if hasattr(config, 'data') else None,
            "save": config.save if hasattr(config, 'save') else None,
            "search": dict(config.stage1),
            "evaluation": dict(config.evaluation) if hasattr(config, 'evaluation') else {}
        })
        
        stage2_config = CfgNode({
            "dataset": config.dataset,
            "search_space": config.search_space if hasattr(config, 'search_space') else None,
            "data": config.data if hasattr(config, 'data') else None,
            "save": config.save if hasattr(config, 'save') else None,
            "search": dict(config.stage2),
            "evaluation": dict(config.evaluation) if hasattr(config, 'evaluation') else {}
        })
        
        # Set stage1 seed from main config if not present
        if hasattr(config.search, 'seed') and not hasattr(stage1_config.search, 'seed'):
            stage1_config.search.seed = config.search.seed
        
        # Set stage2 seed from main config if not present
        if hasattr(config.search, 'seed') and not hasattr(stage2_config.search, 'seed'):
            stage2_config.search.seed = config.search.seed
        
        # Add ZCP specific parameters to stage1
        stage1_config.search.zc = True
        stage1_config.search.use_zc_api = True
        stage1_config.search.zc_names = [self.zcp_method]
        stage1_config.search.zc_only = True
        
        # Create optimizers
        self.stage1_use_zcp = True
        self.stage2_use_zcp = True
        
        logger.info("Using Inverted BANANAS for stage 1")
        zc_api = get_zc_benchmark_api(search_space=config.search_space, dataset=config.dataset)
        self.stage1_optimizer = Inverted_Bananas(stage1_config, zc_api=zc_api)
            
        logger.info("Using ZCP-enhanced GSparsity for stage 2")
        self.stage2_optimizer = ZCP_GSparseOptimizer(stage2_config)
            
        # Track current stage
        self.current_stage = 1
        self.reduced_search_space = None
        self.search_space = None
        self.scope = None
        self.dataset_api = None
        self.train_loader = None
        self.worst_architectures = []
        
        # Store these config values for the Trainer to access
        config.search.stage1_epochs = self.stage1_epochs
        config.search.stage2_epochs = self.stage2_epochs
        config.search.stage1_use_zcp = self.stage1_use_zcp
        config.search.stage2_use_zcp = self.stage2_use_zcp

    def adapt_search_space(self, search_space, scope=None, dataset_api=None, train_loader=None, **kwargs):
        """
        Initialize the search space for both stages
        """
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        self.train_loader = train_loader
        
        # Initialize stage 1 optimizer
        logger.info("Initializing stage 1: Inverted BANANAS")
        self.stage1_optimizer.adapt_search_space(self.search_space, dataset_api=dataset_api)
        
        # Stage 2 will be initialized after stage 1 completes
        return self.search_space

    def step(self, data_train, data_val):
        """
        Delegate the step function to the appropriate stage optimizer
        """
        if self.current_stage == 1:
            # Stage 1 uses a dummy step
            return self.stage1_optimizer.step(data_train, data_val)
        else:
            # Stage 2 uses GSparsity's actual step function
            return self.stage2_optimizer.step(data_train, data_val)

    def new_epoch(self, epoch):
        """
        Handle epoch transitions and stage-specific processing
        """
        # Check if we need to transition from stage 1 to stage 2
        if self.current_stage == 1 and epoch >= self.stage1_epochs:
            logger.info(f"Transitioning to stage 2 at epoch {epoch}")
            self._transition_to_stage2()
            self.current_stage = 2
        
        # Pass the epoch to the current optimizer
        if self.current_stage == 1:
            # Stage 1 optimizer can just handle epochs normally
            self.stage1_optimizer.new_epoch(epoch)
        else:
            # For stage 2, we need to tell it which epoch within stage 2 it is
            stage2_epoch = epoch - self.stage1_epochs
            logger.info(f"=== Stage 2 - Epoch {stage2_epoch} of {self.stage2_epochs} ===")
            
            # Let stage2 optimizer know which epoch it's at
            self.stage2_optimizer.new_epoch(stage2_epoch)
            
            # CRITICAL FIX: Process all batches for this epoch if dataloaders are available
            if hasattr(self, 'train_queue') and hasattr(self, 'valid_queue'):
                self._process_all_batches()

    def _process_all_batches(self):
        """
        Process all batches in the dataloader manually for GSparseOptimizer
        """
        device = next(self.stage2_optimizer.graph.parameters()).device
        logger.info(f"Processing all training batches for epoch ({len(self.train_queue)} batches)")
        
        # Set model to training mode
        self.stage2_optimizer.graph.train()
        
        # Process all batches in the train queue
        for batch_idx, data_train in enumerate(self.train_queue):
            # Get validation batch (cycling if needed)
            try:
                data_val = next(self.valid_iter)
            except (StopIteration, AttributeError):
                self.valid_iter = iter(self.valid_queue)
                data_val = next(self.valid_iter)
            
            # Move data to correct device
            data_train = (data_train[0].to(device), data_train[1].to(device))
            data_val = (data_val[0].to(device), data_val[1].to(device))
            
            # Process batch through the GSparseOptimizer
            self.stage2_optimizer.step(data_train, data_val)
            
            # Log progress periodically
            if batch_idx % 10 == 0:
                logger.info(f"Processed batch {batch_idx}/{len(self.train_queue)}")
        
        logger.info(f"Completed processing all {len(self.train_queue)} batches")

    def _transition_to_stage2(self):
        """
        Transition from stage 1 to stage 2 with proper dataloader reinitialization
        """
        logger.info("Transitioning from stage 1 to stage 2")
        
        # Get the worst architectures from stage 1
        self.worst_architectures = self._get_worst_architectures()
        
        # Save the worst architectures for reference
        self._save_worst_architectures()
        
        # Create reduced search space by removing the worst architectures
        self.reduced_search_space = self._create_reduced_search_space()
        
        # Verify the search space reduction
        self._verify_search_space_reduction(self.search_space, self.reduced_search_space)
        
        # Important: Set the dataset API attribute on the reduced search space
        if hasattr(self.search_space, 'dataset_api'):
            self.reduced_search_space.dataset_api = self.search_space.dataset_api
        
        # Initialize stage 2 with the reduced search space
        if self.config.search.stage2_use_zcp:
            logger.info("Adapting search space for ZCP GSparseOptimizer")
            self.stage2_optimizer.adapt_search_space(
                self.reduced_search_space, 
                scope=self.scope,
                dataset_api=self.dataset_api,
                train_loader=self.train_loader
            )
        else:
            logger.info("Adapting search space for standard GSparseOptimizer")
            self.stage2_optimizer.adapt_search_space(
                self.reduced_search_space,
                scope=self.scope,
                dataset_api=self.dataset_api
            )
        
        # Ensure graph is moved to the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Moving graph to device: {device}")
        self.stage2_optimizer.graph = self.stage2_optimizer.graph.to(device)
        
        # Force proper initialization of the stage 2 optimizer
        logger.info("Calling before_training for stage 2 optimizer")
        self.stage2_optimizer.before_training()
        
        # Store dataset_api in the stage 2 optimizer for later use
        self.stage2_optimizer.dataset_api = self.dataset_api
        
        # CRITICAL FIX: Signal to the trainer that dataloaders need to be rebuilt
        self._force_dataloader_rebuild = True
        
        logger.info(f"Stage 2 initialized with reduced search space")
        self._apply_blacklist_hook_to_gsparsity(self.reduced_search_space)

    def should_rebuild_dataloaders(self):
        """
        Tell the trainer whether it should rebuild dataloaders.
        This is needed when transitioning from stage 1 to stage 2.
        """
        rebuild = getattr(self, '_force_dataloader_rebuild', False)
        # Reset flag after it's been checked once
        if rebuild:
            self._force_dataloader_rebuild = False
        return rebuild

    def _get_worst_architectures(self):
        """Extract the worst architectures from stage 1 results"""
        # Get all evaluated architectures from stage 1
        evaluated_archs = self.stage1_optimizer.history
        
        # Log all evaluated architectures for debugging
        logger.info(f"Stage 1 evaluated {len(evaluated_archs)} total architectures")
        for i, arch in enumerate(evaluated_archs):
            arch_spec = None
            if hasattr(arch.arch, 'get_op_indices'):
                arch_spec = tuple(arch.arch.get_op_indices())
            logger.info(f"Arch {i}: accuracy {arch.accuracy:.4f}, hash {arch_spec}")
        
        if len(evaluated_archs) <= 1:
            # If there's only one or no architectures, can't remove any
            logger.warning("Not enough architectures to remove any (need at least 2)")
            return []
            
        # Sort by accuracy (worst first)
        sorted_archs = sorted(evaluated_archs, key=lambda x: x.accuracy)
        logger.info(f"Worst architecture has accuracy: {sorted_archs[0].accuracy:.4f}")
        logger.info(f"Best architecture has accuracy: {sorted_archs[-1].accuracy:.4f}")
        
        # Determine how many to remove, ensuring at least 1
        num_to_remove = max(1, int(len(sorted_archs) * self.removal_percentage))
        
        # Don't remove all architectures
        if num_to_remove >= len(sorted_archs):
            num_to_remove = len(sorted_archs) - 1
            logger.warning(f"Adjusted removal count to {num_to_remove} to avoid removing all architectures")
            
        worst_archs = sorted_archs[:num_to_remove]
        logger.info(f"Selected {num_to_remove} worst architectures out of {len(sorted_archs)}")
        
        # Add detailed logging about the removed architectures
        for i, arch in enumerate(worst_archs):
            arch_spec = "unknown"
            if hasattr(arch.arch, 'get_op_indices'):
                arch_spec = tuple(arch.arch.get_op_indices())
            logger.info(f"Removing arch {i}: accuracy {arch.accuracy:.4f}, hash {arch_spec}")
        
        return worst_archs

    def _save_worst_architectures(self):
        """
        Save the worst architectures for reference and analysis
        """
        output_path = Path(self.config.save) / "worst_architectures.json"
        
        # Extract relevant information about the worst architectures
        worst_arch_data = []
        for arch in self.worst_architectures:
            arch_data = {
                "accuracy": float(arch.accuracy),
                "genotype": str(arch.arch),
                # Add any other relevant arch info here
            }
            worst_arch_data.append(arch_data)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(worst_arch_data, f, indent=2)
        
        logger.info(f"Saved worst architectures to {output_path}")

    def _create_reduced_search_space(self):
        """
        Create a search space that excludes the worst architectures
        """
        # Create a clone of the original search space
        reduced_space = self.search_space.clone()
        
        # If no architectures to exclude, return the original space
        if not self.worst_architectures:
            logger.info("No architectures to exclude")
            return reduced_space
        
        # Extract architecture specifications from worst architectures
        worst_arch_specs = []
        for arch in self.worst_architectures:
            spec = None
            # For NASBench201, get the operation indices
            if hasattr(arch.arch, 'get_op_indices') and callable(arch.arch.get_op_indices):
                spec = tuple(arch.arch.get_op_indices())
            # For NASBench301, get the compact representation
            elif hasattr(arch.arch, 'get_compact') and callable(arch.arch.get_compact):
                spec = arch.get_compact()
            # Fallback: use string representation
            else:
                spec = str(arch.arch)
            
            worst_arch_specs.append(spec)
            logger.info(f"Blacklisting architecture: {spec}")
        
        # Store the blacklist in a new attribute
        reduced_space.blacklisted_archs = worst_arch_specs
        
        # For NASBench201, directly modify the op_indices selection method
        # This is critical for GSparsity since it modifies edge operations
        if hasattr(reduced_space, 'get_type') and reduced_space.get_type() == 'nasbench201':
            # Get the original op_indices method from the search space
            original_sample = reduced_space.sample_random_architecture
            
            def patched_sample(self, dataset_api=None, **kwargs):
                """Sample architectures excluding blacklisted ones"""
                max_attempts = 100
                for attempt in range(max_attempts):
                    # Create a fresh clone for each attempt to avoid corruption
                    temp_space = self.__class__()
                    temp_space.__dict__.update(self.__dict__)
                    
                    # Call original sampling method on the temporary space
                    original_sample(dataset_api=dataset_api, **kwargs)
                    
                    # We need to parse the architecture to ensure all ops are properly initialized
                    if hasattr(self, 'parse'):
                        self.parse()
                        
                    try:
                        # Now try to get the op indices
                        if hasattr(self, 'get_op_indices'):
                            current_arch = tuple(self.get_op_indices())
                            if current_arch not in self.blacklisted_archs:
                                logger.debug(f"Found non-blacklisted architecture on attempt {attempt+1}: {current_arch}")
                                return
                            else:
                                logger.debug(f"Sampled a blacklisted architecture on attempt {attempt+1}, retrying...")
                    except Exception as e:
                        logger.warning(f"Error during architecture sampling (attempt {attempt+1}): {e}")
                        continue
                        
                logger.warning(f"Could not find non-blacklisted architecture after {max_attempts} attempts")
            
            # Replace the method on the actual search space instance
            reduced_space.sample_random_architecture = patched_sample.__get__(reduced_space, reduced_space.__class__)
            
            # Additionally, add a wrapper around GSparseOptimizer's discretization
            # to ensure it doesn't select blacklisted architectures
            if hasattr(self.stage2_optimizer, 'get_final_architecture'):
                original_get_final = self.stage2_optimizer.get_final_architecture
                
                def patched_get_final(self):
                    """Ensure final architecture isn't blacklisted"""
                    arch = original_get_final()
                    
                    # Check if the architecture is blacklisted
                    if hasattr(arch, 'get_op_indices'):
                        arch_spec = tuple(arch.get_op_indices())
                        if arch_spec in reduced_space.blacklisted_archs:
                            logger.warning("GSparseOptimizer selected a blacklisted architecture! Selecting another one.")
                            # Try a few random architectures
                            for _ in range(10):
                                new_arch = reduced_space.clone()
                                new_arch.sample_random_architecture(dataset_api=self.dataset_api)
                                if tuple(new_arch.get_op_indices()) not in reduced_space.blacklisted_archs:
                                    return new_arch
                    
                    return arch
                    
                # Apply the patched method
                self.stage2_optimizer.get_final_architecture = patched_get_final.__get__(self.stage2_optimizer, self.stage2_optimizer.__class__)
        
        logger.info(f"Created reduced search space excluding {len(worst_arch_specs)} architectures")
        return reduced_space

    def _get_arch_spec(self, arch):
        """
        Extract a canonical specification from an architecture that can be used for comparison.
        The format depends on the search space type.
        
        Args:
            arch: Architecture object from which to extract the specification
        
        Returns:
            A hashable representation of the architecture for comparison
        """
        # Check if it's a NASBench201 architecture
        if hasattr(arch, 'get_op_indices') and callable(arch.get_op_indices):
            # For NB201, get the operation indices as a tuple
            return tuple(arch.get_op_indices())
        
        # Check if it's a NASBench301 architecture
        elif hasattr(arch, 'get_compact') and callable(arch.get_compact):
            # For NB301, get the compact representation
            return arch.get_compact()
        
        # If it's already a compact representation
        elif isinstance(arch, (tuple, list)) or (hasattr(arch, '__iter__') and not isinstance(arch, str)):
            return tuple(arch) if not isinstance(arch, tuple) else arch
            
        # Default case: convert to string for comparison
        return str(arch)

    def before_training(self):
        """Pre-training setup"""
        if self.current_stage == 1:
            self.stage1_optimizer.before_training()
        else:
            self.stage2_optimizer.before_training()

    def after_training(self):
        """Post-training cleanup with validation"""
        if self.current_stage == 2:
            # Get the final architecture and validate it's not blacklisted
            final_arch = self.stage2_optimizer.get_final_architecture()
            
            logger.info("=== FINAL ARCHITECTURE VALIDATION ===")
            logger.info(f"Architecture: {final_arch}")
            
            # Check if it's in the blacklist
            arch_spec = None
            if hasattr(final_arch, 'get_op_indices'):
                arch_spec = tuple(final_arch.get_op_indices())
                
                if hasattr(self.reduced_search_space, 'blacklisted_archs'):
                    if arch_spec in self.reduced_search_space.blacklisted_archs:
                        logger.error(f"❌ CRITICAL: Final architecture {arch_spec} is in blacklist!")
                    else:
                        logger.info(f"✓ Final architecture {arch_spec} is not in blacklist")
            
            # Print architecture details and performance metrics if available
            try:
                val_acc = final_arch.query(Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api)
                test_acc = final_arch.query(Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api)
                logger.info(f"Final architecture validation accuracy: {val_acc:.4f}")
                logger.info(f"Final architecture test accuracy: {test_acc:.4f}")
                
                if hasattr(self, 'worst_architectures') and self.worst_architectures:
                    worst_acc = self.worst_architectures[0].accuracy
                    logger.info(f"Worst removed architecture had accuracy: {worst_acc:.4f}")
                    logger.info(f"Improvement over worst: {val_acc - worst_acc:.4f}")
            except Exception as e:
                logger.error(f"Error querying architecture metrics: {e}")
                
            self.stage2_optimizer.after_training()
        else:
            logger.warning("Training ended during stage 1, no final architecture available")

    def get_final_architecture(self):
        """Get the final architecture from stage 2"""
        if self.current_stage == 2:
            return self.stage2_optimizer.get_final_architecture()
        else:
            logger.error("Cannot get final architecture before stage 2 completes")
            # Return a random valid architecture as fallback
            return self.search_space.clone().sample_random_architecture()

    def get_op_optimizer(self):
        """Get the operation optimizer from stage 2"""
        return self.stage2_optimizer.get_op_optimizer()

    def get_checkpointables(self):
        """Get all objects to save in checkpoints"""
        checkpointables = {
            "current_stage": self.current_stage,
            "worst_architectures": self.worst_architectures,
        }
        
        # Add objects from the current stage
        if self.current_stage == 1:
            stage1_checkpointables = self.stage1_optimizer.get_checkpointables()
            for key, val in stage1_checkpointables.items():
                checkpointables[f"stage1_{key}"] = val
            
            # Add the required "model" key - the trainer expects this
            if "model" in stage1_checkpointables:
                checkpointables["model"] = stage1_checkpointables["model"]
            else:
                # Fallback - use the graph or search space
                checkpointables["model"] = getattr(self.stage1_optimizer, "graph", self.search_space)
        else:
            stage2_checkpointables = self.stage2_optimizer.get_checkpointables()
            for key, val in stage2_checkpointables.items():
                checkpointables[f"stage2_{key}"] = val
            
            # Add the required "model" key
            if "model" in stage2_checkpointables:
                checkpointables["model"] = stage2_checkpointables["model"]
            else:
                # Fallback - use the graph or search space
                checkpointables["model"] = getattr(self.stage2_optimizer, "graph", self.search_space)
        
        return checkpointables

    @property
    def op_optimizer(self):
        """Return the appropriate optimizer based on current stage"""
        if self.current_stage == 1:
            # Inverted_Bananas doesn't have op_optimizer, so we provide a dummy one
            return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        else:
            # GSparsity has an op_optimizer
            return self.stage2_optimizer.op_optimizer

    @property
    def using_step_function(self):
        """
        Dynamically return whether the current stage optimizer uses the step function.
        This ensures the Trainer will use the right evaluation approach for each stage.
        """
        if hasattr(self, '_skip_dataloader_init') and self._skip_dataloader_init:
            # Force not using dataloaders in stage 1
            return False
        
        if self.current_stage == 1:
            return False  # Stage 1 (Inverted Bananas) doesn't use step function
        else:
            return True   # Stage 2 (GSparsity) uses step function

    def __getattr__(self, name):
        """
        Delegate attribute access to the current active optimizer or config
        This helps handle attributes that might be expected by the Trainer
        """
        # First check if the current stage optimizer has the attribute
        current_optimizer = self.stage1_optimizer if self.current_stage == 1 else self.stage2_optimizer
        
        if hasattr(current_optimizer, name):
            return getattr(current_optimizer, name)
        
        # Check if it's in the main config search section
        if hasattr(self.config.search, name):
            return getattr(self.config.search, name)
        
        # Check if it's in the current stage config search section
        current_stage_config = self.config.stage1 if self.current_stage == 1 else self.config.stage2
        if hasattr(current_stage_config.search, name):
            return getattr(current_stage_config.search, name)
        
        # Try the other optimizer as a last resort
        other_optimizer = self.stage2_optimizer if self.current_stage == 1 else self.stage1_optimizer
        if hasattr(other_optimizer, name):
            return getattr(other_optimizer, name)
            
        # If we get here, the attribute doesn't exist
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def train_statistics(self, report_incumbent=True):
        """
        Return training statistics based on the current active stage
        
        Args:
            report_incumbent: Whether to report the overall best architecture or the most recently evaluated one
        
        Returns:
            tuple: (train_accuracy, valid_accuracy, test_accuracy, train_time)
        """
        if self.current_stage == 1:
            # Delegate to stage 1 optimizer
            return self.stage1_optimizer.train_statistics(report_incumbent)
        else:
            # Delegate to stage 2 optimizer
            return self.stage2_optimizer.train_statistics(report_incumbent)

    def before_search(self):
        """
        Called by the trainer before search begins.
        This ensures proper dataloader setup based on current stage.
        """
        # If we're starting in stage 1 (discrete search with inverted bananas)
        # we should explicitly set using_step_function to False to prevent
        # the trainer from expecting dataloaders
        if self.current_stage == 1:
            # This is a special method to tell the trainer not to use dataloaders
            # for the first stage even if using_step_function property is True
            self._skip_dataloader_init = True

    def set_dataloaders(self, train_queue, valid_queue):
        """
        Store references to dataloaders built by the trainer
        """
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.valid_iter = iter(valid_queue)

    def _apply_blacklist_hook_to_gsparsity(self, reduced_space):
        """Apply hooks with enhanced validation"""
        logger.info(f"Setting up blacklist hooks for search space with {len(getattr(reduced_space, 'blacklisted_archs', []))} blacklisted architectures")
        
        # Hook the get_final_architecture method to check against blacklist
        original_get_final = self.stage2_optimizer.get_final_architecture
        
        def patched_get_final(optimizer_self):
            # Get the architecture from the original method
            architecture = original_get_final()
            
            # If we have a blacklist, check against it
            if hasattr(reduced_space, 'blacklisted_archs') and reduced_space.blacklisted_archs:
                # Get the architecture specification
                if hasattr(architecture, 'get_op_indices'):
                    arch_spec = tuple(architecture.get_op_indices())
                    
                    # Detailed logging of selected architecture
                    logger.info(f"Stage 2 selected architecture: {arch_spec}")
                    
                    # Check if it's blacklisted
                    if arch_spec in reduced_space.blacklisted_archs:
                        logger.warning(f"⚠️ GSparseOptimizer selected a blacklisted architecture: {arch_spec}")
                        logger.warning("Attempting to select a non-blacklisted alternative...")
                        
                        # Try to find a non-blacklisted architecture
                        new_arch = reduced_space.clone()
                        new_arch.sample_random_architecture(dataset_api=optimizer_self.dataset_api)
                        
                        # Keep trying until we get a non-blacklisted one
                        attempts = 0
                        while tuple(new_arch.get_op_indices()) in reduced_space.blacklisted_archs:
                            attempts += 1
                            if attempts % 10 == 0:
                                logger.info(f"Made {attempts} attempts to find non-blacklisted architecture")
                            new_arch.sample_random_architecture(dataset_api=optimizer_self.dataset_api)
                            if attempts >= 100:
                                logger.error("Could not find non-blacklisted architecture after 100 attempts")
                                break
                        
                        if attempts < 100:
                            new_arch_spec = tuple(new_arch.get_op_indices())
                            logger.info(f"Found non-blacklisted architecture: {new_arch_spec} after {attempts+1} attempts")
                        return new_arch
                    else:
                        logger.info(f"✓ Selected architecture is not in blacklist: {arch_spec}")
            
            return architecture
        
        # Apply our patched method
        self.stage2_optimizer.get_final_architecture = patched_get_final.__get__(
            self.stage2_optimizer
        )
        
        logger.info("Successfully applied blacklist hooks to GSparseOptimizer")

    def _verify_search_space_reduction(self, original_space, reduced_space):
        """Verify that the search space has been properly reduced"""
        if not hasattr(reduced_space, 'blacklisted_archs') or not reduced_space.blacklisted_archs:
            logger.warning("No blacklisted architectures found in reduced space!")
            return
            
        # For search spaces with deterministic sizes (like NASBench201)
        if hasattr(original_space, 'get_search_space_size') and hasattr(reduced_space, 'get_search_space_size'):
            try:
                orig_size = original_space.get_search_space_size()
                reduced_size = orig_size - len(reduced_space.blacklisted_archs)
                logger.info(f"Search space reduced from {orig_size} to approximately {reduced_size} architectures")
                logger.info(f"Reduction percentage: {len(reduced_space.blacklisted_archs)/orig_size*100:.2f}%")
            except Exception as e:
                logger.warning(f"Could not calculate search space sizes: {e}")
        
        # Simply verify the blacklist exists and has entries
        logger.info(f"Blacklist verification: {len(reduced_space.blacklisted_archs)} architectures blacklisted")
        for i, arch_spec in enumerate(reduced_space.blacklisted_archs):
            logger.info(f"  Blacklisted arch {i}: {arch_spec}")