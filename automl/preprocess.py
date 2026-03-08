"""
================================================================================
PREPROCESSING MODULE - Weather Forecasting AutoML System
================================================================================

PURPOSE:
    Clean data, engineer features, and prepare data for machine learning models.
    This is the MOST IMPORTANT module for forecasting accuracy!

FEATURE CATEGORIES:
    1. Temporal Features  - Hour, day, month, season (time patterns)
    2. Lag Features       - Past values (what was temp 1h, 6h, 24h ago?)
    3. Rolling Features   - Moving averages/statistics over windows
    4. Difference Features - Rate of change (how fast is temp changing?)

USAGE:
    from automl.preprocess import WeatherPreprocessor
    
    preprocessor = WeatherPreprocessor()
    X, y, features, targets = preprocessor.prepare_for_training(df, horizon=24)

AUTHOR: Deepika Jakka
DATE: March 2026
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd                              # DataFrames for tabular data
import numpy as np                               # Numerical operations
from typing import List, Optional, Tuple, Dict, Any  # Type hints
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Feature scaling
import logging                                   # Status logging

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(level=logging.INFO)          # Show INFO and above
logger = logging.getLogger(__name__)             # Logger for this module


# =============================================================================
# WEATHER PREPROCESSOR CLASS
# =============================================================================
class WeatherPreprocessor:
    """
    Comprehensive preprocessing pipeline for weather time series data.
    
    This class handles all data preparation steps:
    - Data cleaning (missing values, outliers)
    - Feature engineering (temporal, lag, rolling, difference features)
    - Target variable creation (shifted values for forecasting)
    - Feature scaling (normalization)
    
    Attributes:
        datetime_col: Name of the datetime column in the data
        target_cols: List of columns we want to predict
        scalers: Dictionary storing fitted scalers for later use
        feature_cols: List of feature column names after preprocessing
    """
    
    # -------------------------------------------------------------------------
    # WEATHER FEATURE COLUMNS
    # -------------------------------------------------------------------------
    # These are the standard weather measurement columns we expect
    WEATHER_FEATURES = [
        'temperature_c',               # Air temperature in Celsius
        'precipitation_mm',            # Rainfall in millimeters
        'wet_bulb_temp_c',             # Wet bulb temperature
        'dew_point_c',                 # Dew point temperature
        'vapour_pressure_hpa',         # Vapour pressure
        'relative_humidity_pct',       # Relative humidity percentage
        'mean_sea_level_pressure_hpa'  # Atmospheric pressure
    ]
    
    # -------------------------------------------------------------------------
    # CONSTRUCTOR
    # -------------------------------------------------------------------------
    def __init__(self, 
                 datetime_col: str = 'datetime',
                 target_cols: Optional[List[str]] = None):
        """
        Initialize the preprocessor with configuration options.
        
        Args:
            datetime_col: Name of the datetime column (default: 'datetime')
            target_cols: List of columns to predict. If None, uses defaults:
                         [temperature_c, precipitation_mm, relative_humidity_pct]
        
        Example:
            # Default usage
            prep = WeatherPreprocessor()
            
            # Custom target
            prep = WeatherPreprocessor(target_cols=['temperature_c'])
        """
        # Store the datetime column name for later use
        self.datetime_col = datetime_col
        
        # Set target columns - these are the variables we want to forecast
        # Default: temperature, precipitation, and humidity
        self.target_cols = target_cols or [
            'temperature_c',           # Main target: air temperature
            'precipitation_mm',        # Secondary: rainfall
            'relative_humidity_pct'    # Secondary: humidity
        ]
        
        # Dictionary to store fitted scalers (for transforming new data later)
        self.scalers: Dict[str, Any] = {}
        
        # Will store feature column names after preprocessing
        self.feature_cols: List[str] = []
        
    # =========================================================================
    # DATA CLEANING
    # =========================================================================
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data by handling missing values and outliers.
        
        Steps performed:
        1. Remove rows with missing datetime (can't use them for time series)
        2. Sort data chronologically
        3. Fill missing values using forward/backward fill (time series method)
        4. Clip extreme outliers using IQR method
        
        Args:
            df: Raw DataFrame to clean
            
        Returns:
            Cleaned DataFrame with no missing values in key columns
            
        Note:
            We use forward-fill then backward-fill because:
            - Weather changes gradually, so nearby values are good estimates
            - This preserves the time series nature of the data
        """
        df = df.copy()  # Don't modify the original DataFrame
        logger.info(f"Cleaning data with {len(df)} rows")
        
        # ---------------------------------------------------------------------
        # STEP 1: Remove rows with missing datetime
        # ---------------------------------------------------------------------
        # We can't use observations without timestamps in time series analysis
        initial_rows = len(df)
        df = df.dropna(subset=[self.datetime_col])  # Drop rows where datetime is NaN
        logger.info(f"Removed {initial_rows - len(df)} rows with missing datetime")
        
        # ---------------------------------------------------------------------
        # STEP 2: Sort chronologically
        # ---------------------------------------------------------------------
        # Time series data MUST be in chronological order for:
        # - Lag features to be correct
        # - Rolling windows to work properly
        # - Train/test split to be valid (no future data leakage)
        df = df.sort_values(self.datetime_col).reset_index(drop=True)
        
        # ---------------------------------------------------------------------
        # STEP 3: Fill missing values in weather columns
        # ---------------------------------------------------------------------
        # Get list of weather columns that actually exist in the data
        weather_cols = [col for col in self.WEATHER_FEATURES if col in df.columns]
        
        for col in weather_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                # Forward fill: use the previous valid value
                # Then backward fill: for any remaining NaN at the start
                df[col] = df[col].ffill()    # Forward fill
                df[col] = df[col].bfill()    # Backward fill remaining
                logger.info(f"Filled {missing_count} missing values in {col}")
        
        # ---------------------------------------------------------------------
        # STEP 4: Handle extreme outliers using IQR method
        # ---------------------------------------------------------------------
        # IQR method: values outside Q1 - 3*IQR to Q3 + 3*IQR are outliers
        # We use 3*IQR (instead of typical 1.5*IQR) to only clip extreme values
        for col in weather_cols:
            if col == 'precipitation_mm':
                # Precipitation is special: can have legitimate zeros and high values
                # Only ensure non-negative (rain can't be negative!)
                df = df[df[col] >= 0]
            else:
                # For other columns, clip extreme outliers
                Q1 = df[col].quantile(0.001)  # 0.1th percentile
                Q3 = df[col].quantile(0.999)  # 99.9th percentile
                IQR = Q3 - Q1                  # Interquartile range
                
                lower = Q1 - 3 * IQR  # Lower bound
                upper = Q3 + 3 * IQR  # Upper bound
                
                # Count outliers before clipping
                outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                
                if outliers > 0:
                    # Clip values to bounds (don't remove, just limit)
                    df[col] = df[col].clip(lower, upper)
                    logger.info(f"Clipped {outliers} outliers in {col}")
        
        logger.info(f"Final cleaned dataset: {len(df)} rows")
        return df
    
    # =========================================================================
    # TEMPORAL FEATURE ENGINEERING
    # =========================================================================
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from the datetime column.
        
        WHY TEMPORAL FEATURES MATTER:
        - Weather has strong seasonal patterns (winter vs summer)
        - Weather has daily patterns (night vs day)
        - These patterns help models predict future values
        
        FEATURES CREATED:
        - hour (0-23): Hour of day
        - day_of_week (0-6): Monday=0, Sunday=6
        - day_of_month (1-31): Day number
        - month (1-12): Month number
        - year: Year
        - day_of_year (1-366): Day number in year
        - week_of_year (1-53): Week number
        - quarter (1-4): Quarter of year
        - hour_sin, hour_cos: Cyclical encoding of hour
        - day_sin, day_cos: Cyclical encoding of day
        - month_sin, month_cos: Cyclical encoding of month
        - is_daytime (0/1): Binary indicator
        - is_weekend (0/1): Binary indicator
        - season (0-3): Winter/Spring/Summer/Fall
        
        Args:
            df: DataFrame with datetime column
            
        Returns:
            DataFrame with additional temporal features
            
        Note on Cyclical Encoding:
            Hour 23 and Hour 0 are only 1 hour apart, but numerically 23.
            Cyclical encoding using sin/cos makes them close in feature space:
            - hour_sin = sin(2π * hour / 24)
            - hour_cos = cos(2π * hour / 24)
        """
        df = df.copy()
        dt = df[self.datetime_col]  # Extract datetime series for convenience
        
        # ---------------------------------------------------------------------
        # BASIC TEMPORAL FEATURES
        # ---------------------------------------------------------------------
        df['hour'] = dt.dt.hour          # Hour of day (0-23)
        df['day_of_week'] = dt.dt.dayofweek  # Day of week (Monday=0, Sunday=6)
        df['day_of_month'] = dt.dt.day   # Day of month (1-31)
        df['month'] = dt.dt.month        # Month (1-12)
        df['year'] = dt.dt.year          # Year
        df['day_of_year'] = dt.dt.dayofyear  # Day number in year (1-366)
        df['week_of_year'] = dt.dt.isocalendar().week.astype(int)  # Week (1-53)
        df['quarter'] = dt.dt.quarter    # Quarter (1-4)
        
        # ---------------------------------------------------------------------
        # CYCLICAL ENCODING
        # ---------------------------------------------------------------------
        # Problem: Hour 23 and Hour 0 are 1 hour apart, but 23 units apart numerically
        # Solution: Map to circle using sin/cos so 23 and 0 are close
        
        # Hour cyclical encoding (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of year cyclical encoding (365-day cycle)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Month cyclical encoding (12-month cycle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # ---------------------------------------------------------------------
        # BINARY INDICATORS
        # ---------------------------------------------------------------------
        # Is it daytime? (rough approximation for Ireland: 6am - 8pm)
        # Temperature and solar radiation behave differently day vs night
        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 20)).astype(int)
        
        # Is it weekend? (Saturday=5, Sunday=6)
        # Human activity patterns differ on weekends (less urban heat island)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # ---------------------------------------------------------------------
        # SEASON INDICATOR
        # ---------------------------------------------------------------------
        # Season is crucial for weather prediction (Northern Hemisphere)
        df['season'] = df['month'].apply(self._get_season)
        
        logger.info("Created temporal features")
        return df
    
    def _get_season(self, month: int) -> int:
        """
        Map month number to season for Northern Hemisphere.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Season code: 0=Winter, 1=Spring, 2=Summer, 3=Fall
        """
        if month in [12, 1, 2]:    # December, January, February
            return 0  # Winter
        elif month in [3, 4, 5]:   # March, April, May
            return 1  # Spring
        elif month in [6, 7, 8]:   # June, July, August
            return 2  # Summer
        else:                       # September, October, November
            return 3  # Fall
    
    # =========================================================================
    # LAG FEATURE ENGINEERING
    # =========================================================================
    def create_lag_features(self, 
                           df: pd.DataFrame,
                           lag_hours: List[int] = [1, 2, 3, 6, 12, 24],
                           target_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create lagged features (past values) for time series prediction.
        
        WHY LAG FEATURES MATTER:
        - Weather is autocorrelated: tomorrow's temp depends on today's
        - The temp 1 hour ago is very similar to now
        - The temp 24 hours ago captures daily patterns
        
        FEATURES CREATED (for each target column):
        - {col}_lag_1h:  Value 1 hour ago
        - {col}_lag_2h:  Value 2 hours ago
        - {col}_lag_3h:  Value 3 hours ago
        - {col}_lag_6h:  Value 6 hours ago
        - {col}_lag_12h: Value 12 hours ago
        - {col}_lag_24h: Value 24 hours ago (same time yesterday)
        
        Args:
            df: DataFrame with weather data
            lag_hours: List of lag periods in hours (default: [1,2,3,6,12,24])
            target_cols: Columns to create lags for (default: self.target_cols)
            
        Returns:
            DataFrame with lag features added
            
        Note:
            Lag features create NaN values at the start of the dataset
            (we don't have data from before the first observation)
        """
        df = df.copy()
        target_cols = target_cols or self.target_cols
        
        # Create lag features for each target column
        for col in target_cols:
            if col not in df.columns:
                continue  # Skip if column doesn't exist
                
            for lag in lag_hours:
                # shift(n) moves values down by n rows
                # So shift(1) gives us the value from 1 row (1 hour) ago
                df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
        
        logger.info(f"Created lag features for {len(lag_hours)} time steps")
        return df
    
    # =========================================================================
    # ROLLING WINDOW FEATURE ENGINEERING
    # =========================================================================
    def create_rolling_features(self,
                                df: pd.DataFrame,
                                windows: List[int] = [3, 6, 12, 24],
                                target_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create rolling window statistics (moving averages, etc.).
        
        WHY ROLLING FEATURES MATTER:
        - Smooth out noise in the data
        - Capture trends over different time periods
        - 24h rolling mean captures the "typical" value for that time of year
        
        FEATURES CREATED (for each window and target):
        - {col}_rolling_mean_{w}h:  Average over past w hours
        - {col}_rolling_std_{w}h:   Std deviation over past w hours
        - {col}_rolling_min_{w}h:   Minimum over past w hours
        - {col}_rolling_max_{w}h:   Maximum over past w hours
        
        Args:
            df: DataFrame with weather data
            windows: Window sizes in hours (default: [3, 6, 12, 24])
            target_cols: Columns to compute rolling stats for
            
        Returns:
            DataFrame with rolling features added
            
        Example:
            temperature_c_rolling_mean_24h = average temp over past 24 hours
        """
        df = df.copy()
        target_cols = target_cols or self.target_cols
        
        for col in target_cols:
            if col not in df.columns:
                continue
                
            for window in windows:
                # Rolling window with min_periods=1 to handle edge cases
                # min_periods=1 means: compute even if window not full
                
                # Rolling mean (average)
                df[f'{col}_rolling_mean_{window}h'] = (
                    df[col].rolling(window, min_periods=1).mean()
                )
                
                # Rolling standard deviation (variability)
                df[f'{col}_rolling_std_{window}h'] = (
                    df[col].rolling(window, min_periods=1).std()
                )
                
                # Rolling minimum
                df[f'{col}_rolling_min_{window}h'] = (
                    df[col].rolling(window, min_periods=1).min()
                )
                
                # Rolling maximum
                df[f'{col}_rolling_max_{window}h'] = (
                    df[col].rolling(window, min_periods=1).max()
                )
        
        logger.info(f"Created rolling features for {len(windows)} window sizes")
        return df
    
    # =========================================================================
    # DIFFERENCE FEATURE ENGINEERING
    # =========================================================================
    def create_difference_features(self,
                                  df: pd.DataFrame,
                                  periods: List[int] = [1, 24],
                                  target_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create difference features (rate of change).
        
        WHY DIFFERENCE FEATURES MATTER:
        - Capture whether temperature is rising or falling
        - Differentiate between stable and changing weather
        - Important for predicting weather fronts
        
        FEATURES CREATED:
        - {col}_diff_1h:  Change from 1 hour ago (hourly rate of change)
        - {col}_diff_24h: Change from 24 hours ago (daily change)
        
        Args:
            df: DataFrame with weather data
            periods: Periods to calculate differences for (default: [1, 24])
            target_cols: Columns to compute differences for
            
        Returns:
            DataFrame with difference features added
            
        Example:
            If temp is 15°C now and was 13°C an hour ago:
            temperature_c_diff_1h = 2.0 (rising 2° per hour)
        """
        df = df.copy()
        target_cols = target_cols or self.target_cols
        
        for col in target_cols:
            if col not in df.columns:
                continue
                
            for period in periods:
                # diff(n) = current value - value n rows ago
                df[f'{col}_diff_{period}h'] = df[col].diff(period)
        
        logger.info(f"Created difference features")
        return df
    
    # =========================================================================
    # TARGET VARIABLE CREATION
    # =========================================================================
    def create_target_variables(self,
                               df: pd.DataFrame,
                               horizons: List[int] = [1, 24]) -> pd.DataFrame:
        """
        Create target variables for forecasting.
        
        HOW FORECASTING TARGETS WORK:
        - We want to predict future values
        - The target is the value `horizon` hours AHEAD
        - We shift values BACKWARD (negative) to align with current features
        
        Example for horizon=24:
        - Row 0 has features from hour 0
        - Target is the value from hour 24 (24 hours later)
        - So target = shift(-24) → move future value to current row
        
        Args:
            df: DataFrame with weather data
            horizons: Forecast horizons in hours (default: [1, 24])
            
        Returns:
            DataFrame with target columns added
            
        Note:
            This creates NaN at the END of the dataset
            (we don't know the future for the last observations)
        """
        df = df.copy()
        
        for col in self.target_cols:
            if col not in df.columns:
                continue
                
            for horizon in horizons:
                # shift(-horizon) moves values UP (future to present)
                # So current row now has the value from `horizon` hours later
                df[f'{col}_target_{horizon}h'] = df[col].shift(-horizon)
        
        logger.info(f"Created target variables for horizons: {horizons}")
        return df
    
    # =========================================================================
    # FEATURE SCALING
    # =========================================================================
    def scale_features(self,
                      df: pd.DataFrame,
                      feature_cols: List[str],
                      method: str = 'standard',
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features for better model performance.
        
        WHY SCALE FEATURES:
        - Many ML algorithms work better with scaled features
        - Features with different scales can dominate others
        - StandardScaler: mean=0, std=1 (works for most algorithms)
        - MinMaxScaler: range [0, 1] (good for neural networks)
        
        Args:
            df: DataFrame with features
            feature_cols: List of column names to scale
            method: 'standard' (z-score) or 'minmax' (0-1 range)
            fit: True = fit and transform, False = use existing scaler
            
        Returns:
            DataFrame with scaled features
            
        Note:
            When fit=True, we save the scaler for later use
            When fit=False, we use the saved scaler (for test data)
        """
        df = df.copy()
        
        if fit:
            # Create and fit a new scaler
            if method == 'standard':
                scaler = StandardScaler()  # z-score: (x - mean) / std
            else:
                scaler = MinMaxScaler()    # (x - min) / (max - min)
            
            # Fit and transform the data
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            
            # Save scaler for later use on test data
            self.scalers['features'] = scaler
        else:
            # Use previously fitted scaler
            if 'features' not in self.scalers:
                raise ValueError("No fitted scaler found. Call with fit=True first.")
            df[feature_cols] = self.scalers['features'].transform(df[feature_cols])
        
        return df
    
    # =========================================================================
    # COMPLETE PREPROCESSING PIPELINE
    # =========================================================================
    def prepare_for_training(self,
                            df: pd.DataFrame,
                            horizon: int = 1,
                            scale: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
        """
        Run the complete preprocessing pipeline: clean → features → targets → scale.
        
        This is the main method to call for preparing data for training.
        
        PIPELINE STEPS:
        1. Clean data (missing values, outliers)
        2. Create temporal features (hour, month, season, etc.)
        3. Create lag features (values from past hours)
        4. Create rolling features (moving averages)
        5. Create difference features (rate of change)
        6. Create target variables (future values to predict)
        7. Drop rows with NaN (from lagging/shifting)
        8. Scale features (standardize)
        
        Args:
            df: Raw weather DataFrame
            horizon: Forecast horizon in hours (1 = 1-hour ahead)
            scale: Whether to standardize features (recommended=True)
            
        Returns:
            Tuple containing:
            - X: DataFrame of feature columns
            - y: DataFrame of target columns
            - feature_names: List of feature column names
            - target_names: List of target column names
            
        Example:
            X, y, features, targets = prep.prepare_for_training(df, horizon=24)
            # X shape: (n_samples, 93)  # ~93 features
            # y shape: (n_samples, 3)   # 3 targets by default
        """
        logger.info("Starting full preprocessing pipeline")
        
        # -----------------------------------------------------------------
        # STEP 1: Clean the data
        # -----------------------------------------------------------------
        df = self.clean_data(df)
        
        # -----------------------------------------------------------------
        # STEP 2-5: Create all features
        # -----------------------------------------------------------------
        df = self.create_temporal_features(df)    # Hour, month, season, etc.
        df = self.create_lag_features(df)          # Past values
        df = self.create_rolling_features(df)      # Moving averages
        df = self.create_difference_features(df)   # Rate of change
        
        # -----------------------------------------------------------------
        # STEP 6: Create target variables
        # -----------------------------------------------------------------
        df = self.create_target_variables(df, horizons=[horizon])
        
        # -----------------------------------------------------------------
        # STEP 7: Drop rows with NaN
        # -----------------------------------------------------------------
        # NaN values come from:
        # - Lag features (no data before first observation)
        # - Target variables (no future data for last observations)
        df = df.dropna()
        logger.info(f"Dataset after dropping NaN: {len(df)} rows")
        
        # -----------------------------------------------------------------
        # IDENTIFY FEATURE AND TARGET COLUMNS
        # -----------------------------------------------------------------
        # Target columns: {col}_target_{horizon}h
        target_names = [
            f'{col}_target_{horizon}h' 
            for col in self.target_cols 
            if f'{col}_target_{horizon}h' in df.columns
        ]
        
        # Feature columns: everything except datetime, targets, and raw target cols
        exclude_cols = [self.datetime_col] + target_names + self.target_cols
        feature_names = [
            col for col in df.columns 
            if col not in exclude_cols 
            and df[col].dtype in ['int64', 'float64', 'int32', 'float32']
        ]
        
        # Store for later reference
        self.feature_cols = feature_names
        
        # -----------------------------------------------------------------
        # EXTRACT X AND Y
        # -----------------------------------------------------------------
        X = df[feature_names].copy()  # Features
        y = df[target_names].copy()   # Targets
        
        # -----------------------------------------------------------------
        # STEP 8: Scale features
        # -----------------------------------------------------------------
        if scale:
            X = self.scale_features(X, feature_names, fit=True)
        
        logger.info(f"Prepared {len(feature_names)} features and {len(target_names)} targets")
        
        return X, y, feature_names, target_names


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================
def preprocess_weather_data(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Convenience function for preprocessing weather data in one call.
    
    This is a shortcut that creates a WeatherPreprocessor and runs the pipeline.
    
    Args:
        df: Raw weather DataFrame
        **kwargs: Arguments passed to prepare_for_training()
                  (e.g., horizon=24, scale=True)
        
    Returns:
        Tuple of (X, y, feature_names, target_names)
        
    Example:
        X, y, features, targets = preprocess_weather_data(df, horizon=1)
    """
    preprocessor = WeatherPreprocessor()
    return preprocessor.prepare_for_training(df, **kwargs)
