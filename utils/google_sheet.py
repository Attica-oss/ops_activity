"""Google Sheets Integration with Railway Programming Pattern"""

from dataclasses import dataclass
from pathlib import Path
from io import StringIO
from typing import Dict
import polars as pl
import requests


import utils.error
from utils.result import Result

try:
    from dotenv import load_dotenv, set_key
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class SheetEnvironmentManager:
    """Handles environment variable operations for Google Sheets"""

    @staticmethod
    def _check_dotenv_availability() -> 'Result[bool]':
        """Check if python-dotenv is available"""
        if not DOTENV_AVAILABLE:
            return Result.err(
                "python-dotenv library is required for .env file support. "
                "Install it with: pip install python-dotenv"
            )
        return Result.ok(True)
    
    @classmethod
    def load_sheet_configs(cls, env_file: str = ".env") -> 'Result[Dict[str, Dict[str, str]]]':
        """
        Load all sheet configurations from .env file
        Returns format: {config_name: {sheet_id: str, sheet_name: str}}
        """
        check_result = cls._check_dotenv_availability()
        if not check_result.success:
            return check_result
   
        try:
            env_path = Path(env_file)
            if not env_path.exists():
                return Result.err(f"File {env_file} does not exist")

            load_dotenv(env_path)
         
            configs = {}
            # Look for patterns like SHEET1_ID, SHEET1_NAME, etc.
            env_vars = {}
         
            with open(env_path, "r",encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        var_name, value = line.split("=", 1)
                        var_name = var_name.strip()
                        value = value.strip().strip("\"'")
                        env_vars[var_name] = value
      
            # Group related sheet configs
            sheet_prefixes = set()
            for var_name in env_vars:
                if var_name.endswith("_ID") or var_name.endswith("_NAME"):
                    prefix = var_name.rsplit("_", 1)[0]
                    sheet_prefixes.add(prefix)
            
            for prefix in sheet_prefixes:
                sheet_id_var = f"{prefix}_ID"
                sheet_name_var = f"{prefix}_NAME"
                
                if sheet_id_var in env_vars and sheet_name_var in env_vars:
                    sheet_id = env_vars[sheet_id_var]
                    sheet_name = env_vars[sheet_name_var]
                    
                    # Validate sheet ID
                    if cls._validate_sheet_id(sheet_id):
                        configs[prefix.lower()] = {
                            "sheet_id": sheet_id,
                            "sheet_name": sheet_name
                        }
            
            return Result.ok(configs)
            
        except Exception as e:
            return Result.err(f"Error reading {env_file}: {str(e)}")
    
    @classmethod
    def save_sheet_config(
        cls, 
        config_name: str,
        sheet_id: str,
        sheet_name: str,
        env_file: str = ".env"
    ) -> 'Result[str]':
        """Save sheet configuration to .env file"""
        check_result = cls._check_dotenv_availability()
        if not check_result.success:
            return check_result
        
        # Validate sheet ID
        if not cls._validate_sheet_id(sheet_id):
            return Result.err(f"Invalid sheet ID: {sheet_id}")
        
        if not sheet_name.strip():
            return Result.err("Sheet name cannot be empty")
        
        try:
            env_path = Path(env_file)
            if not env_path.exists():
                env_path.touch()
            
            config_upper = config_name.upper()
            sheet_id_var = f"{config_upper}_ID"
            sheet_name_var = f"{config_upper}_NAME"
            
            # Save both variables
            success1 = set_key(str(env_path), sheet_id_var, sheet_id)
            success2 = set_key(str(env_path), sheet_name_var, sheet_name)
            
            if success1 and success2:
                return Result.ok(f"Sheet config '{config_name}' saved to {env_file}")
            else:
                return Result.err(f"Failed to save sheet config to {env_file}")
                
        except Exception as e:
            return Result.err(f"Error writing to {env_file}: {str(e)}")
    
    @classmethod
    def get_sheet_config(cls, config_name: str, env_file: str = ".env") -> 'Result[Dict[str, str]]':
        """Get a specific sheet configuration"""
        configs_result = cls.load_sheet_configs(env_file)
        if not configs_result.success:
            return configs_result
        
        configs = configs_result.data
        config_key = config_name.lower()
        
        if config_key in configs:
            return Result.ok(configs[config_key])
        else:
            return Result.err(f"Sheet config '{config_name}' not found in {env_file}")
    
    @staticmethod
    def _validate_sheet_id(sheet_id: str) -> bool:
        """Validate Google Sheet ID format"""
        return (
            isinstance(sheet_id, str) and
            len(sheet_id.strip()) > 0 and
            len(sheet_id) == 44 and
            all(c.isalnum() or c in "-_" for c in sheet_id)
        )


@dataclass
class GoogleSheetConfig:
    """Configuration for loading a Google Sheet"""
    sheet_id: str
    sheet_name: str
    timeout: int = 10
    
    def _post_init_(self):
        """Validate configuration after initialization"""
        validation_result = self._validate()
        if not validation_result.success:
            raise error.ConfigurationError(validation_result.error)
    
    def _validate(self) -> 'Result[bool]':
        """Validate the configuration"""
        if not SheetEnvironmentManager._validate_sheet_id(self.sheet_id):
            return Result.err(f"Invalid sheet ID: {self.sheet_id}")
        
        if not isinstance(self.sheet_name, str) or not self.sheet_name.strip():
            return Result.err("Sheet name must be a non-empty string")
        
        if self.timeout <= 0:
            return Result.err("Timeout must be positive")
        
        return Result.ok(True)
    
    def create_url(self) -> 'Result[str]':
        """Create URL for Google Sheets CSV export"""
        base_url = "https://docs.google.com/spreadsheets/d/"
        url = f"{base_url}{self.sheet_id}/gviz/tq?tqx=out:csv&sheet={self.sheet_name}"
        return Result.ok(url)
    
    def fetch_data(self) -> 'Result[StringIO]':
        """Fetch data from Google Sheets"""
        def make_request(url: str) -> StringIO:
            try:
                response = requests.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    return StringIO(response.text)
                else:
                    raise error.SheetFetchError(
                        f"Failed to fetch data. Status: {response.status_code}, "
                        f"Reason: {response.reason}"
                    )
            except requests.exceptions.Timeout:
                raise error.SheetFetchError(
                    f"Request timed out after {self.timeout} seconds"
                )
            except requests.exceptions.RequestException as e:
                raise error.NetworkError(f"Network error: {str(e)}")
        
        return self.create_url().bind(lambda url: Result.ok(make_request(url)))
    
    def to_lazyframe(self, parse_dates: bool = True) -> 'Result[pl.LazyFrame]':
        """Load data as Polars LazyFrame"""
        def create_lazyframe(csv_data: StringIO) -> pl.LazyFrame:
            try:
                return pl.read_csv(csv_data, try_parse_dates=parse_dates).lazy()
            except Exception as e:
                raise error.SheetTransformError(f"Failed to parse CSV data: {str(e)}")
        
        return self.fetch_data().bind(lambda data: Result.ok(create_lazyframe(data)))
    
    def to_dataframe(self, parse_dates: bool = True) -> 'Result[pl.DataFrame]':
        """Load data as Polars DataFrame"""
        return self.to_lazyframe(parse_dates).map(lambda lf: lf.collect())


class GoogleSheetsLoader:
    """High-level interface for loading Google Sheets"""
    
    @staticmethod
    def from_env(config_name: str, env_file: str = ".env") -> 'Result[GoogleSheetConfig]':
        """Create GoogleSheetConfig from environment configuration"""
        config_result = SheetEnvironmentManager.get_sheet_config(config_name, env_file)
        if not config_result.success:
            return config_result
        
        config = config_result.data
        try:
            return Result.ok(GoogleSheetConfig(
                sheet_id=config["sheet_id"],
                sheet_name=config["sheet_name"]
            ))
        except Exception as e:
            return Result.err(f"Failed to create config: {str(e)}")
    
    @staticmethod
    def load_sheet(
        config_name: str, 
        env_file: str = ".env",
        as_dataframe: bool = False,
        parse_dates: bool = True
    ) -> 'Result[pl.LazyFrame]':
        """Load a Google Sheet directly from environment config"""
        def load_data(config: GoogleSheetConfig):
            if as_dataframe:
                return config.to_dataframe(parse_dates)
            else:
                return config.to_lazyframe(parse_dates)
        
        return (GoogleSheetsLoader.from_env(config_name, env_file)
                .bind(load_data))
    
    @staticmethod
    def list_available_sheets(env_file: str = ".env") -> Result:
        """List all available sheet configurations"""
        return SheetEnvironmentManager.load_sheet_configs(env_file)
    
    @staticmethod
    def save_sheet_config(
        config_name: str,
        sheet_id: str, 
        sheet_name: str,
        env_file: str = ".env"
    ) -> 'Result[str]':
        """Save a new sheet configuration"""
        return SheetEnvironmentManager.save_sheet_config(
            config_name, sheet_id, sheet_name, env_file
        )


# Convenience function for marimo notebooks
def load_google_sheet(
    config_name: str,
    env_file: str = ".env",
    as_dataframe: bool = False,
    parse_dates: bool = True
) -> 'Result[pl.LazyFrame]':
    """
    Convenience function to load Google Sheets in marimo notebooks
    
    Args:
        config_name: Name of the configuration in .env (e.g., 'SALES' for SALES_ID and SALES_NAME)
        env_file: Path to .env file
        as_dataframe: Return DataFrame instead of LazyFrame
        parse_dates: Attempt to parse dates
    
    Returns:
        Result containing LazyFrame/DataFrame or error
    """
    return GoogleSheetsLoader.load_sheet(config_name, env_file, as_dataframe, parse_dates)