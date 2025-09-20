"""Result Type for Railway Programming Pattern"""
from dataclasses import dataclass
from typing import Optional, Any, Callable, TypeVar, Generic

T = TypeVar('T')
U = TypeVar('U')


@dataclass
class Result(Generic[T]):
    """Result type for railway programming pattern with improved type hints"""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, data: T) -> 'Result[T]':
        """Create a successful result"""
        return cls(success=True, data=data)

    @classmethod
    def err(cls, error: str) -> 'Result[Any]':
        """Create an error result"""
        return cls(success=False, error=error)

    def bind(self, func: Callable[[T], 'Result[U]']) -> 'Result[U]':
        """Monadic bind operation for chaining operations"""
        if not self.success:
            return Result.err(self.error)
        try:
            return func(self.data)
        except Exception as e:
            return Result.err(str(e))

    def map(self, func: Callable[[T], U]) -> 'Result[U]':
        """Map function over successful result"""
        if not self.success:
            return Result.err(self.error)
        try:
            return Result.ok(func(self.data))
        except Exception as e:
            return Result.err(str(e))

    def unwrap(self) -> T:
        """Unwrap the result data, raising an exception if error"""
        if not self.success:
            raise ValueError(f"Result contains error: {self.error}")
        return self.data

    def unwrap_or(self, default: T) -> T:
        """Unwrap the result data, returning default if error"""
        return self.data if self.success else default

    def is_ok(self) -> bool:
        """Check if result is successful"""
        return self.success

    def is_err(self) -> bool:
        """Check if result is an error"""
        return not self.success