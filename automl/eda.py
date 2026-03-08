"""
================================================================================
EXPLORATORY DATA ANALYSIS (EDA) MODULE - Weather Forecasting AutoML
================================================================================

PURPOSE:
    Automatically analyze weather data to understand patterns, quality issues,
    and relationships before building ML models.

KEY ANALYSES:
    - Basic Statistics: mean, std, min, max, missing values, skewness
    - Temporal Coverage: date range, gaps, data completeness
    - Correlations: relationships between weather variables
    - Seasonal Patterns: monthly, hourly, and weekly patterns
    - Outlier Detection: identify anomalous values

USAGE:
    from automl.eda import WeatherEDA
    
    eda = WeatherEDA(df)           # Create EDA object
    eda.print_summary()            # Print human-readable summary
    report = eda.generate_report() # Get full report as dictionary

AUTHOR: Deepika Jakka
DATE: March 2026
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd   # DataFrame operations
import numpy as np    # Numerical computing
from typing import Dict, Any, List, Optional, Tuple  # Type hints
import logging        # Status logging

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(level=logging.INFO)  # Configure log level
logger = logging.getLogger(__name__)     # Get logger for this module


# =============================================================================
# WEATHER EDA CLASS
# =============================================================================
class WeatherEDA:
    """
    Automated exploratory data analysis for weather time series data.
    
    This class provides methods to:
    - Calculate descriptive statistics
    - Analyze temporal coverage and detect gaps
    - Compute correlation matrices
    - Identify seasonal patterns (monthly, hourly, daily)
    - Detect outliers using IQR or Z-score methods
    - Generate comprehensive reports
    
    Attributes:
        df: The weather DataFrame to analyze
        datetime_col: Name of the datetime column
        numeric_cols: List of numeric column names
    """
    
    def __init__(self, df: pd.DataFrame, datetime_col: str = 'datetime'):
        """
        Initialize EDA with the dataset to analyze.
        
        Args:
            df: Weather DataFrame containing observations
            datetime_col: Name of the datetime column (default: 'datetime')
        """
        self.df = df.copy()                # Store a copy to avoid modifying original
        self.datetime_col = datetime_col   # Store datetime column name
        # Find all numeric columns for analysis
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
    def basic_statistics(self) -> pd.DataFrame:
        """
        Calculate basic descriptive statistics for all numeric columns.
        
        Returns:
            DataFrame with statistics: count, mean, std, min, max, quartiles,
            missing count, missing %, skewness, kurtosis
        """
        if not self.numeric_cols:
            logger.warning("No numeric columns found for statistics")
            return pd.DataFrame()
        
        # Get pandas describe() output
        stats = self.df[self.numeric_cols].describe()
        
        # Add missing value statistics
        stats.loc['missing'] = self.df[self.numeric_cols].isnull().sum()
        stats.loc['missing_pct'] = (self.df[self.numeric_cols].isnull().sum() / len(self.df) * 100).round(2)
        
        # Add distribution shape statistics
        stats.loc['skewness'] = self.df[self.numeric_cols].skew()
        stats.loc['kurtosis'] = self.df[self.numeric_cols].kurtosis()
        
        return stats
    
    def temporal_coverage(self) -> Dict[str, Any]:
        """
        Analyze the temporal coverage of the dataset.
        
        Returns:
            Dictionary with: start_date, end_date, duration, coverage %
        """
        if self.datetime_col not in self.df.columns:
            return {'error': 'No datetime column found'}
        
        dt = self.df[self.datetime_col]
        
        # Calculate expected vs actual observations for hourly data
        date_range = (dt.max() - dt.min()).total_seconds() / 3600
        expected_obs = int(date_range) + 1
        actual_obs = len(self.df)
        
        return {
            'start_date': dt.min(),
            'end_date': dt.max(),
            'duration_days': (dt.max() - dt.min()).days,
            'duration_years': round((dt.max() - dt.min()).days / 365.25, 2),
            'expected_observations': expected_obs,
            'actual_observations': actual_obs,
            'coverage_pct': round(actual_obs / expected_obs * 100, 2) if expected_obs > 0 else 0,
            'missing_hours': expected_obs - actual_obs
        }
    
    def detect_gaps(self, max_gap_hours: int = 24) -> pd.DataFrame:
        """
        Detect gaps (missing periods) larger than threshold in the time series.
        
        Args:
            max_gap_hours: Only report gaps larger than this (default: 24)
            
        Returns:
            DataFrame with gap_start, gap_end, duration_hours, missing_observations
        """
        if self.datetime_col not in self.df.columns:
            return pd.DataFrame()
        
        df_sorted = self.df.sort_values(self.datetime_col)
        time_diff = df_sorted[self.datetime_col].diff()
        
        # Find gaps larger than threshold
        gaps = time_diff[time_diff > pd.Timedelta(hours=max_gap_hours)]
        
        gap_info = []
        for idx in gaps.index:
            gap_start = df_sorted.loc[idx - 1, self.datetime_col] if idx > 0 else None
            gap_end = df_sorted.loc[idx, self.datetime_col]
            gap_duration = gaps.loc[idx]
            
            gap_info.append({
                'gap_start': gap_start,
                'gap_end': gap_end,
                'duration_hours': gap_duration.total_seconds() / 3600,
                'missing_observations': int(gap_duration.total_seconds() / 3600)
            })
        
        return pd.DataFrame(gap_info)
    
    def correlation_analysis(self, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix for all numeric features.
        
        Args:
            target_col: If provided, sort by correlation with this column
            
        Returns:
            Correlation matrix as DataFrame
        """
        corr_matrix = self.df[self.numeric_cols].corr()
        
        if target_col and target_col in corr_matrix.columns:
            corr_with_target = corr_matrix[target_col].abs().sort_values(ascending=False)
            corr_matrix = corr_matrix.loc[corr_with_target.index, corr_with_target.index]
        
        return corr_matrix
    
    def seasonal_analysis(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze seasonal patterns: monthly, hourly, and weekly.
        
        Returns:
            Dictionary with 'monthly', 'hourly', 'daily' pattern DataFrames
        """
        if self.datetime_col not in self.df.columns:
            return {}
        
        df = self.df.copy()
        df['month'] = df[self.datetime_col].dt.month
        df['hour'] = df[self.datetime_col].dt.hour
        df['day_of_week'] = df[self.datetime_col].dt.dayofweek
        
        weather_cols = [col for col in ['temperature_c', 'precipitation_mm', 'relative_humidity_pct'] 
                       if col in self.numeric_cols]
        
        if not weather_cols:
            weather_cols = self.numeric_cols[:3]
        
        return {
            'monthly': df.groupby('month')[weather_cols].agg(['mean', 'std', 'min', 'max']),
            'hourly': df.groupby('hour')[weather_cols].agg(['mean', 'std']),
            'daily': df.groupby('day_of_week')[weather_cols].agg(['mean', 'std'])
        }
    
    def outlier_analysis(self, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Dict[str, Any]]:
        """
        Detect outliers in each numeric column using IQR or Z-score method.
        
        Args:
            method: 'iqr' (default) or 'zscore'
            threshold: Multiplier for IQR (1.5) or Z-score threshold (3.0)
            
        Returns:
            Dictionary with outlier statistics per column
        """
        outlier_stats = {}
        
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = data[(data < lower_bound) | (data > upper_bound)]
            else:  # zscore
                mean = data.mean()
                std = data.std()
                z_scores = np.abs((data - mean) / std)
                outliers = data[z_scores > threshold]
            
            outlier_stats[col] = {
                'n_outliers': len(outliers),
                'pct_outliers': round(len(outliers) / len(data) * 100, 2),
                'outlier_min': outliers.min() if len(outliers) > 0 else None,
                'outlier_max': outliers.max() if len(outliers) > 0 else None
            }
        
        return outlier_stats
    
    def feature_importance_preview(self, target_col: str) -> pd.DataFrame:
        """
        Quick feature importance estimate using correlation with target.
        
        Args:
            target_col: The target variable column name
            
        Returns:
            DataFrame with features sorted by absolute correlation
        """
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        feature_cols = [col for col in self.numeric_cols if col != target_col]
        
        importance = []
        for col in feature_cols:
            corr = self.df[col].corr(self.df[target_col])
            importance.append({
                'feature': col,
                'correlation': corr,
                'abs_correlation': abs(corr)
            })
        
        return pd.DataFrame(importance).sort_values('abs_correlation', ascending=False)
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive EDA report as a dictionary.
        
        Returns:
            Dictionary containing all EDA results (JSON-serializable)
        """
        logger.info("Generating EDA report...")
        
        def df_to_dict(df):
            """Convert DataFrame to JSON-safe dict, handling MultiIndex."""
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
            return df.to_dict()
        
        basic_stats = self.basic_statistics()
        correlations = self.correlation_analysis()
        gaps = self.detect_gaps()
        seasonal = self.seasonal_analysis()
        
        report = {
            'basic_stats': df_to_dict(basic_stats),
            'temporal_coverage': self.temporal_coverage(),
            'gaps': df_to_dict(gaps) if len(gaps) > 0 else None,
            'correlations': df_to_dict(correlations),
            'seasonal_patterns': {k: df_to_dict(v) for k, v in seasonal.items()},
            'outliers': self.outlier_analysis()
        }
        
        logger.info("EDA report generated")
        return report
    
    def print_summary(self) -> None:
        """Print a human-readable summary of the EDA to console."""
        print("\n" + "="*60)
        print("WEATHER DATA EXPLORATORY ANALYSIS")
        print("="*60)
        
        print(f"\nDataset Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"Numeric Columns: {len(self.numeric_cols)}")
        
        coverage = self.temporal_coverage()
        if 'error' not in coverage:
            print(f"\nTemporal Coverage:")
            print(f"  Date Range: {coverage['start_date']} to {coverage['end_date']}")
            print(f"  Duration: {coverage['duration_years']} years ({coverage['duration_days']} days)")
            print(f"  Coverage: {coverage['coverage_pct']}%")
        
        if self.numeric_cols:
            stats = self.basic_statistics()
            if not stats.empty:
                print(f"\nMissing Values:")
                for col in self.numeric_cols[:5]:
                    missing = stats.loc['missing_pct', col] if col in stats.columns else 0
                    print(f"  {col}: {missing}%")
            
            outliers = self.outlier_analysis()
            if outliers:
                print(f"\nOutlier Summary (IQR method):")
                for col, col_stats in list(outliers.items())[:5]:
                    print(f"  {col}: {col_stats['n_outliers']} outliers ({col_stats['pct_outliers']}%)")
        
        print("\n" + "="*60)


# ------------------------------------------------------------------------
# CONVENIENCE FUNCTION
# ------------------------------------------------------------------------

def run_eda(df: pd.DataFrame, datetime_col: str = 'datetime', save_report: bool = True) -> Dict[str, Any]:
    """Quick helper that runs EDA and optionally returns the report.

    Args:
        df: DataFrame to analyze.
        datetime_col: Name of datetime column to use.
        save_report: If True, returns the generated report dictionary.

    Returns:
        The report dictionary if `save_report` is True, else an empty dict.
    """
    eda = WeatherEDA(df, datetime_col=datetime_col)
    eda.print_summary()
    report = eda.generate_report()
    if save_report:
        return report
    return {}
