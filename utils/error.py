"""Custom Error Classes for Google Sheets Operations"""

class GoogleSheetError(Exception):
    """Base exception for Google Sheets operations"""

class InvalidSheetIDError(GoogleSheetError):
    """Raised when sheet ID is invalid"""


class SheetFetchError(GoogleSheetError):
    """Raised when fetching sheet data fails"""


class SheetTransformError(GoogleSheetError):
    """Raised when transforming sheet data fails"""

class EnvError(GoogleSheetError):
    """Raised when environment operations fail"""


class ConfigurationError(GoogleSheetError):
    """Raised when configuration is invalid"""


class NetworkError(GoogleSheetError):
    """Raised when network operations fail"""


class ValidationError(GoogleSheetError):
    """Raised when data validation fails"""