import torch
import gc
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def get_gpu_memory_info() -> Tuple[float, float]:
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0.0, 0.0

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def find_optimal_batch_size(model, sample_batch, initial_batch_size: int = 32, 
                          min_batch_size: int = 1, safety_factor: float = 0.9) -> int:
    """
    Find the optimal batch size that fits in GPU memory.
    
    Args:
        model: The model to test
        sample_batch: A sample batch dict with input_ids, attention_mask, etc.
        initial_batch_size: Starting batch size to try
        min_batch_size: Minimum acceptable batch size
        safety_factor: Use only this fraction of max batch size for safety
        
    Returns:
        Optimal batch size that fits in memory
    """
    if not torch.cuda.is_available():
        return initial_batch_size
    
    # Clear cache before testing
    clear_gpu_memory()
    
    batch_size = initial_batch_size
    max_working_batch_size = min_batch_size
    
    # Get single sample size
    single_sample = {k: v[:1] for k, v in sample_batch.items() if torch.is_tensor(v)}
    
    while batch_size >= min_batch_size:
        try:
            # Clear memory before each test
            clear_gpu_memory()
            
            # Create test batch of current size
            test_batch = {}
            for k, v in single_sample.items():
                if len(v.shape) > 1:
                    # Repeat along batch dimension
                    test_batch[k] = v.repeat(batch_size, *([1] * (len(v.shape) - 1)))
                else:
                    test_batch[k] = v.repeat(batch_size)
            
            # Try forward pass
            with torch.no_grad():
                _ = model(**test_batch)
            
            # If successful, update max working batch size
            max_working_batch_size = batch_size
            logger.info(f"Batch size {batch_size} fits in memory")
            
            # Try larger batch size
            batch_size = int(batch_size * 1.5)
            
        except torch.cuda.OutOfMemoryError:
            logger.info(f"Batch size {batch_size} caused OOM")
            # Try smaller batch size
            batch_size = int(batch_size * 0.5)
            clear_gpu_memory()
        except Exception as e:
            logger.error(f"Error testing batch size {batch_size}: {str(e)}")
            batch_size = int(batch_size * 0.5)
    
    # Apply safety factor
    optimal_batch_size = max(min_batch_size, int(max_working_batch_size * safety_factor))
    logger.info(f"Optimal batch size: {optimal_batch_size}")
    
    return optimal_batch_size

class DynamicBatchSizeCallback:
    """Callback to dynamically adjust batch size on OOM."""
    
    def __init__(self, min_batch_size: int = 1, reduction_factor: float = 0.5):
        self.min_batch_size = min_batch_size
        self.reduction_factor = reduction_factor
        self.current_batch_size = None
        
    def on_oom(self, trainer, current_batch_size: int) -> Optional[int]:
        """
        Called when OOM occurs. Returns new batch size or None to stop.
        """
        new_batch_size = max(self.min_batch_size, int(current_batch_size * self.reduction_factor))
        
        if new_batch_size < current_batch_size:
            logger.warning(f"Reducing batch size from {current_batch_size} to {new_batch_size} due to OOM")
            clear_gpu_memory()
            return new_batch_size
        else:
            logger.error(f"Cannot reduce batch size below {self.min_batch_size}")
            return None