"""
================================================================================
DATA LOADER MODULE - Weather Forecasting AutoML System
================================================================================

PURPOSE:
    Load and parse meteorological data from Irish Met Éireann CSV files.
    The CSV format has a special structure with metadata headers.

USAGE:
    from automl.data_loader import WeatherDataLoader
    
    loader = WeatherDataLoader('hly175.csv')  # Create loader instance
    df = loader.load()                         # Load data into DataFrame
    info = loader.get_station_info()          # Get station metadata

DATA FORMAT:
    Irish Met Éireann files have ~15 header lines with metadata,
    then data starting with 'date,ind,rain,ind.1,temp,...'
    
AUTHOR: Deepika Jakka
DATE: March 2026
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd       # Data manipulation library - handles DataFrames
import numpy as np        # Numerical computing library - handles arrays
from pathlib import Path  # Modern path handling - works across OS
from typing import Union, Optional, Dict, Any  # Type hints for documentation
import logging            # Logging for debugging and status messages

# =============================================================================
# LOGGING SETUP
# =============================================================================
# Configure logging to show INFO level and above (INFO, WARNING, ERROR, CRITICAL)
logging.basicConfig(level=logging.INFO)
# Create logger for this module - uses module name for identification
logger = logging.getLogger(__name__)


# =============================================================================
# WEATHER DATA LOADER CLASS
# =============================================================================
class WeatherDataLoader:
    """
    Load and parse weather data from Irish Met Éireann CSV files.
    
    This class handles:
    - Parsing header metadata (station name, location, height)
    - Finding where actual data starts in the file
    - Loading data into a pandas DataFrame
    - Renaming cryptic column names to descriptive ones
    - Converting columns to correct data types
    
    Attributes:
        data_path: Path to the CSV file
        station_info: Dictionary containing station metadata
        raw_data: The loaded DataFrame (None until load() is called)
    """
    
    # -------------------------------------------------------------------------
    # COLUMN NAME MAPPING
    # -------------------------------------------------------------------------
    # The Irish Met Éireann CSV uses short cryptic column names
    # We map them to descriptive names for clarity
    COLUMN_MAPPING = {
        'date': 'datetime',                    # Observation timestamp
        'rain': 'precipitation_mm',            # Rainfall in millimeters
        'temp': 'temperature_c',               # Air temperature in Celsius
        'wetb': 'wet_bulb_temp_c',             # Wet bulb temperature (humidity measure)
        'dewpt': 'dew_point_c',                # Dew point temperature
        'vappr': 'vapour_pressure_hpa',        # Water vapour pressure in hectopascals
        'rhum': 'relative_humidity_pct',       # Relative humidity as percentage
        'msl': 'mean_sea_level_pressure_hpa'   # Sea level pressure in hectopascals
    }
    
    # -------------------------------------------------------------------------
    # CONSTRUCTOR
    # -------------------------------------------------------------------------
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize the data loader with a path to the CSV file.
        
        Args:
            data_path: Path to the CSV data file (can be string or Path object)
            
        Example:
            loader = WeatherDataLoader('hly175.csv')
            loader = WeatherDataLoader(Path('/data/weather.csv'))
        """
        # Convert to Path object for consistent handling across operating systems
        self.data_path = Path(data_path)
        
        # Dictionary to store station metadata extracted from file header
        self.station_info: Dict[str, Any] = {}
        
        # Will hold the loaded DataFrame after load() is called
        self.raw_data: Optional[pd.DataFrame] = None
        
    # -------------------------------------------------------------------------
    # PARSE FILE HEADER
    # -------------------------------------------------------------------------
    def _parse_header(self, file_path: Path) -> int:
        """
        Parse the file header to extract station metadata and find where data begins.
        
        Irish Met Éireann CSV files have a special format:
        - First ~15 lines contain metadata (station name, location, etc.)
        - Actual data starts at the line beginning with 'date,'
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Row number (0-indexed) where actual data begins
            
        Side Effect:
            Populates self.station_info with extracted metadata
        """
        header_lines = []  # Store header lines for parsing
        data_start = 0     # Row where data starts
        
        # Read file line by line to find where data starts
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                # Data starts at line beginning with 'date,'
                if line.startswith('date,'):
                    data_start = i
                    break  # Found data start, stop reading
                # Store header lines for metadata extraction
                header_lines.append(line.strip())
        
        # Parse station information from header lines
        for line in header_lines:
            # Extract station name
            if 'Station Name:' in line:
                self.station_info['name'] = line.split(':')[1].strip()
            # Extract station height above sea level
            elif 'Station Height:' in line:
                self.station_info['height'] = line.split(':')[1].strip()
            # Extract latitude and longitude (on same line, comma-separated)
            elif 'Latitude:' in line:
                parts = line.split(',')  # Split "Latitude: 53.35, Longitude: -6.27"
                self.station_info['latitude'] = float(parts[0].split(':')[1])
                self.station_info['longitude'] = float(parts[1].split(':')[1])
                
        return data_start
    
    # -------------------------------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------------------------------
    def load(self, 
             parse_dates: bool = True,      # Convert date strings to datetime?
             drop_indicators: bool = True,  # Remove quality indicator columns?
             rename_columns: bool = True    # Use descriptive column names?
             ) -> pd.DataFrame:
        """
        Load the weather data from CSV file into a pandas DataFrame.
        
        This method:
        1. Parses the header to find where data starts
        2. Loads the CSV data (skipping header lines)
        3. Optionally drops indicator columns (data quality flags)
        4. Optionally parses the date column as datetime
        5. Optionally renames columns to descriptive names
        6. Converts all numeric columns to proper numeric types
        
        Args:
            parse_dates: If True, convert 'date' column to datetime type
            drop_indicators: If True, remove 'ind' columns (quality indicators)
            rename_columns: If True, rename columns using COLUMN_MAPPING
            
        Returns:
            DataFrame containing the weather observations
            
        Example:
            df = loader.load()
            df = loader.load(parse_dates=False)  # Keep dates as strings
        """
        # Log what we're doing
        logger.info(f"Loading data from {self.data_path}")
        
        # Step 1: Parse header and find where data starts
        header_row = self._parse_header(self.data_path)
        
        # Step 2: Load the CSV, skipping metadata header rows
        df = pd.read_csv(
            self.data_path, 
            skiprows=header_row,  # Skip metadata lines
            low_memory=False      # Load entire file to determine types
        )
        
        # Log loading status
        logger.info(f"Loaded {len(df):,} observations")
        logger.info(f"Station: {self.station_info.get('name', 'Unknown')}")
        
        # Step 3: Drop indicator columns (they mark data quality, not needed for ML)
        if drop_indicators:
            # Find columns named 'ind' or 'ind.1', 'ind.2', etc.
            ind_cols = [col for col in df.columns 
                       if col == 'ind' or col.startswith('ind.')]
            df = df.drop(columns=ind_cols, errors='ignore')  # Ignore if not found
            logger.info(f"Dropped {len(ind_cols)} indicator columns")
        
        # Step 4: Parse dates into datetime objects for time series analysis
        if parse_dates and 'date' in df.columns:
            # Format: "01-Jan-2003 00:00" → Python datetime
            df['date'] = pd.to_datetime(
                df['date'], 
                format='%d-%b-%Y %H:%M',  # Day-Month-Year Hour:Minute
                errors='coerce'           # Invalid dates become NaT (Not a Time)
            )
            # Sort chronologically and reset index
            df = df.sort_values('date').reset_index(drop=True)
        
        # Step 5: Rename columns to descriptive names
        if rename_columns:
            df = df.rename(columns=self.COLUMN_MAPPING)
        
        # Step 6: Convert all non-date columns to numeric type
        # Some values may be strings like ' ' or 'NA'
        numeric_cols = [col for col in df.columns if col not in ['datetime', 'date']]
        for col in numeric_cols:
            # to_numeric with errors='coerce' converts invalid values to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Store the loaded data for later access
        self.raw_data = df
        return df
    
    # -------------------------------------------------------------------------
    # ACCESSOR METHODS
    # -------------------------------------------------------------------------
    def get_station_info(self) -> Dict[str, Any]:
        """
        Return the station metadata extracted from file header.
        
        Returns:
            Dictionary with keys: 'name', 'latitude', 'longitude', 'height'
        """
        return self.station_info
    
    def get_date_range(self) -> tuple:
        """
        Return the date range (start, end) of the loaded data.
        
        Returns:
            Tuple of (earliest_date, latest_date) as datetime objects
            
        Raises:
            ValueError: If data has not been loaded yet
        """
        # Check that data has been loaded
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load() first.")
        
        # Determine which column name was used for dates
        date_col = 'datetime' if 'datetime' in self.raw_data.columns else 'date'
        
        # Return min and max dates
        return (
            self.raw_data[date_col].min(),
            self.raw_data[date_col].max()
        )
    
    # -------------------------------------------------------------------------
    # SUMMARY METHOD
    # -------------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """
        Return a comprehensive summary of the loaded data.
        
        Returns:
            Dictionary containing station info, data shape, date range, etc.
            
        Raises:
            ValueError: If data has not been loaded yet
        """
        # Check that data has been loaded
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load() first.")
            
        return {
            'station': self.station_info,
            'n_observations': len(self.raw_data),
            'date_range': self.get_date_range(),
            'columns': list(self.raw_data.columns),
            'missing_values': self.raw_data.isnull().sum().to_dict(),
            'memory_usage_mb': self.raw_data.memory_usage(deep=True).sum() / 1024**2
        }


def load_weather_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Convenience function to load weather data.
    
    Args:
        file_path: Path to the CSV file.
        **kwargs: Additional arguments passed to WeatherDataLoader.load()
        
    Returns:
        DataFrame with the weather data.
    """
    loader = WeatherDataLoader(file_path)
    return loader.load(**kwargs)
