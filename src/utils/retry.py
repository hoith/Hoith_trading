import time
import logging
from functools import wraps
from typing import Callable, Type, Tuple, Any
from alpaca.common.exceptions import APIError

logger = logging.getLogger(__name__)


def retry_on_exception(max_retries: int = 3, delay: float = 1.0, 
                      exponential_backoff: bool = True,
                      exceptions: Tuple[Type[Exception], ...] = (APIError, ConnectionError, TimeoutError)):
    """Decorator to retry function calls on specific exceptions."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        raise e
                    
                    # Calculate delay
                    current_delay = delay
                    if exponential_backoff:
                        current_delay = delay * (2 ** attempt)
                    
                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}: {e}. "
                        f"Retrying in {current_delay:.1f} seconds..."
                    )
                    
                    time.sleep(current_delay)
                except Exception as e:
                    # Don't retry on unexpected exceptions
                    logger.error(f"Function {func.__name__} failed with unexpected exception: {e}")
                    raise e
            
            # This shouldn't be reached, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


class RetryableOperation:
    """Context manager for retryable operations."""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0,
                 exponential_backoff: bool = True,
                 exceptions: Tuple[Type[Exception], ...] = (APIError, ConnectionError)):
        self.max_retries = max_retries
        self.delay = delay
        self.exponential_backoff = exponential_backoff
        self.exceptions = exceptions
        self.attempt = 0
        self.last_exception = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, self.exceptions):
            self.last_exception = exc_val
            
            if self.attempt < self.max_retries:
                # Calculate delay
                current_delay = self.delay
                if self.exponential_backoff:
                    current_delay = self.delay * (2 ** self.attempt)
                
                logger.warning(
                    f"Operation failed on attempt {self.attempt + 1}/{self.max_retries + 1}: {exc_val}. "
                    f"Retrying in {current_delay:.1f} seconds..."
                )
                
                time.sleep(current_delay)
                self.attempt += 1
                return True  # Suppress the exception
            else:
                logger.error(f"Operation failed after {self.max_retries + 1} attempts: {exc_val}")
                return False  # Let the exception propagate
        
        return False  # Don't suppress other exceptions


def with_retry(operation: Callable, *args, max_retries: int = 3, 
               delay: float = 1.0, **kwargs) -> Any:
    """Execute an operation with retry logic."""
    for attempt in range(max_retries + 1):
        try:
            return operation(*args, **kwargs)
        except (APIError, ConnectionError, TimeoutError) as e:
            if attempt == max_retries:
                logger.error(f"Operation failed after {max_retries + 1} attempts: {e}")
                raise e
            
            current_delay = delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay:.1f}s...")
            time.sleep(current_delay)


class CircuitBreaker:
    """Circuit breaker pattern for API calls."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = 'half_open'
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is OPEN - too many failures")
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset if we were in half-open state
            if self.state == 'half_open':
                self.state = 'closed'
                self.failure_count = 0
                logger.info("Circuit breaker reset to closed state")
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
            
            raise e
    
    def reset(self):
        """Manually reset the circuit breaker."""
        self.state = 'closed'
        self.failure_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset")


# Global circuit breaker instance for API calls
api_circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)