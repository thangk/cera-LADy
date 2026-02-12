import torch
import gc
from typing import Dict, Any, Optional
from argparse import Namespace
import logging

logger = logging.getLogger(__name__)

def train_with_dynamic_batching(train_func, args: Namespace) -> Any:
    """
    Wrapper to train BERT with dynamic batch size adjustment on OOM.
    
    Args:
        train_func: The original training function
        args: Training arguments namespace
        
    Returns:
        The trained model
    """
    original_batch_size = args.per_gpu_train_batch_size
    current_batch_size = original_batch_size
    min_batch_size = getattr(args, 'min_batch_size', 1)
    reduction_factor = getattr(args, 'batch_reduction_factor', 0.5)
    
    attempt = 0
    max_attempts = 5
    
    while attempt < max_attempts and current_batch_size >= min_batch_size:
        try:
            # Update batch size in args
            args.per_gpu_train_batch_size = current_batch_size
            logger.info(f"Attempting to train with batch size: {current_batch_size}")
            
            # Clear GPU memory before attempt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Try training
            model = train_func(args)
            logger.info(f"Training successful with batch size: {current_batch_size}")
            return model
            
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"OOM with batch size {current_batch_size}: {str(e)}")
            
            # Reduce batch size
            current_batch_size = max(min_batch_size, int(current_batch_size * reduction_factor))
            
            if current_batch_size < args.per_gpu_train_batch_size:
                logger.info(f"Reducing batch size to {current_batch_size}")
                attempt += 1
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    # Wait a bit for memory to be freed
                    import time
                    time.sleep(2)
            else:
                logger.error(f"Cannot reduce batch size below {min_batch_size}")
                raise
                
        except Exception as e:
            logger.error(f"Non-OOM error during training: {str(e)}")
            raise
    
    # If we get here, we've exhausted all attempts on GPU
    # Try one final attempt on CPU
    if not args.no_cuda and torch.cuda.is_available():
        logger.warning("All GPU attempts failed. Trying on CPU...")
        args.no_cuda = True
        args.per_gpu_train_batch_size = min_batch_size
        
        try:
            model = train_func(args)
            logger.info("Training successful on CPU")
            return model
        except Exception as e:
            logger.error(f"CPU training also failed: {str(e)}")
    
    raise RuntimeError(f"Failed to train after {max_attempts} attempts with minimum batch size {min_batch_size}")

def create_gradient_accumulation_steps(original_batch_size: int, current_batch_size: int, 
                                     current_accumulation_steps: int = 1) -> int:
    """
    Calculate gradient accumulation steps to maintain effective batch size.
    
    Args:
        original_batch_size: The desired effective batch size
        current_batch_size: The actual batch size that fits in memory
        current_accumulation_steps: Current gradient accumulation steps
        
    Returns:
        New gradient accumulation steps
    """
    if current_batch_size >= original_batch_size:
        return current_accumulation_steps
    
    # Calculate how many accumulation steps we need
    effective_batch_size = original_batch_size * current_accumulation_steps
    new_accumulation_steps = max(1, effective_batch_size // current_batch_size)
    
    logger.info(f"Adjusting gradient accumulation steps from {current_accumulation_steps} to {new_accumulation_steps}")
    logger.info(f"Effective batch size: {current_batch_size * new_accumulation_steps}")
    
    return new_accumulation_steps