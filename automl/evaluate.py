"""
Evaluation Module for Weather Forecasting AutoML System.

Provides comprehensive model evaluation metrics and comparison tools.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherModelEvaluator:
    """Evaluate weather forecasting models with multiple metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.evaluation_results: Dict[str, Dict] = {}
        
    def calculate_metrics(self,
                         y_true: Union[np.ndarray, pd.DataFrame],
                         y_pred: Union[np.ndarray, pd.DataFrame],
                         target_name: str = 'target') -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            target_name: Name of the target variable.
            
        Returns:
            Dictionary of metrics.
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE - handle zero values
        if np.any(y_true == 0):
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
        else:
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # Explained variance
        evs = explained_variance_score(y_true, y_pred)
        
        # Additional metrics
        # Skill score (compared to persistence baseline)
        persistence_pred = np.roll(y_true, 1)
        persistence_pred[0] = y_true[0]
        persistence_mse = mean_squared_error(y_true, persistence_pred)
        skill_score = 1 - (mse / persistence_mse) if persistence_mse > 0 else 0
        
        # Mean Bias Error
        mbe = np.mean(y_pred - y_true)
        
        # Correlation coefficient
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        return {
            'target': target_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'explained_variance': evs,
            'skill_score': skill_score,
            'mean_bias_error': mbe,
            'correlation': correlation,
            'n_samples': len(y_true)
        }
    
    def evaluate_model(self,
                      model: Any,
                      X_test: Union[np.ndarray, pd.DataFrame],
                      y_test: Union[np.ndarray, pd.DataFrame],
                      model_name: str = 'model',
                      target_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate a model on test data.
        
        Args:
            model: Trained model with predict method.
            X_test: Test features.
            y_test: Test targets.
            model_name: Name of the model.
            target_names: Names of target variables.
            
        Returns:
            Evaluation results.
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_pred = model.predict(X_test_np)
        
        # Convert to numpy arrays
        y_test_np = y_test.values if isinstance(y_test, pd.DataFrame) else y_test
        
        # Ensure both are 2D for consistent handling
        if y_test_np.ndim == 1:
            y_test_np = y_test_np.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        n_targets = y_test_np.shape[1]
        target_names = target_names or [f'target_{i}' for i in range(n_targets)]
        
        # Calculate metrics for each target
        results = {
            'model_name': model_name,
            'targets': {}
        }
        
        overall_metrics = []
        for i, target_name in enumerate(target_names[:n_targets]):
            metrics = self.calculate_metrics(y_test_np[:, i], y_pred[:, i], target_name)
            results['targets'][target_name] = metrics
            overall_metrics.append(metrics)
        
        # Calculate aggregate metrics
        results['aggregate'] = {
            'mean_rmse': np.mean([m['rmse'] for m in overall_metrics]),
            'mean_mae': np.mean([m['mae'] for m in overall_metrics]),
            'mean_r2': np.mean([m['r2'] for m in overall_metrics]),
            'mean_skill_score': np.mean([m['skill_score'] for m in overall_metrics])
        }
        
        self.evaluation_results[model_name] = results
        
        logger.info(f"  {model_name}: R² = {results['aggregate']['mean_r2']:.4f}, "
                   f"RMSE = {results['aggregate']['mean_rmse']:.4f}")
        
        return results
    
    def compare_models(self,
                      models: Dict[str, Any],
                      X_test: Union[np.ndarray, pd.DataFrame],
                      y_test: Union[np.ndarray, pd.DataFrame],
                      target_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of model_name -> model.
            X_test: Test features.
            y_test: Test targets.
            target_names: Names of target variables.
            
        Returns:
            DataFrame comparing model performance.
        """
        comparison = []
        
        for model_name, model in models.items():
            try:
                results = self.evaluate_model(model, X_test, y_test, model_name, target_names)
                comparison.append({
                    'model': model_name,
                    'mean_rmse': results['aggregate']['mean_rmse'],
                    'mean_mae': results['aggregate']['mean_mae'],
                    'mean_r2': results['aggregate']['mean_r2'],
                    'skill_score': results['aggregate']['mean_skill_score']
                })
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                comparison.append({
                    'model': model_name,
                    'error': str(e)
                })
        
        df = pd.DataFrame(comparison)
        if 'mean_r2' in df.columns:
            df = df.sort_values('mean_r2', ascending=False, na_position='last')
        
        return df
    
    def get_predictions(self,
                       model: Any,
                       X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get predictions from a model.
        
        Args:
            model: Trained model.
            X: Features to predict on.
            
        Returns:
            Predictions array.
        """
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        return model.predict(X_np)
    
    def residual_analysis(self,
                         y_true: Union[np.ndarray, pd.DataFrame],
                         y_pred: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze prediction residuals.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            Residual analysis results.
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        residuals = y_true - y_pred
        
        return {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'min_residual': np.min(residuals),
            'max_residual': np.max(residuals),
            'median_residual': np.median(residuals),
            'skewness': pd.Series(residuals).skew(),
            'kurtosis': pd.Series(residuals).kurtosis(),
            'pct_over_prediction': (residuals < 0).sum() / len(residuals) * 100,
            'pct_under_prediction': (residuals > 0).sum() / len(residuals) * 100
        }
    
    def forecast_accuracy_by_horizon(self,
                                    model: Any,
                                    X_test: pd.DataFrame,
                                    y_test: pd.DataFrame,
                                    horizons: List[int] = [1, 6, 12, 24]) -> pd.DataFrame:
        """
        Analyze how accuracy degrades with forecast horizon.
        
        Note: This requires multiple target columns for different horizons.
        
        Args:
            model: Trained model (or dict of models by horizon).
            X_test: Test features.
            y_test: Test targets with horizon-specific columns.
            horizons: Forecast horizons to analyze.
            
        Returns:
            DataFrame with accuracy by horizon.
        """
        results = []
        
        for horizon in horizons:
            horizon_cols = [col for col in y_test.columns if f'_{horizon}h' in col]
            if horizon_cols:
                for col in horizon_cols:
                    if col in y_test.columns:
                        y_true = y_test[col].values
                        if isinstance(model, dict):
                            if horizon in model:
                                y_pred = model[horizon].predict(X_test)
                            else:
                                continue
                        else:
                            y_pred = model.predict(X_test.values)
                            if y_pred.ndim > 1:
                                idx = list(y_test.columns).index(col)
                                y_pred = y_pred[:, idx] if idx < y_pred.shape[1] else y_pred[:, 0]
                        
                        metrics = self.calculate_metrics(y_true, y_pred, col)
                        results.append({
                            'horizon': horizon,
                            'target': col,
                            'rmse': metrics['rmse'],
                            'mae': metrics['mae'],
                            'r2': metrics['r2']
                        })
        
        return pd.DataFrame(results)
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Evaluation report dictionary.
        """
        if not self.evaluation_results:
            return {'error': 'No models evaluated yet'}
        
        # Find best model
        best_model = None
        best_r2 = -np.inf
        
        for model_name, results in self.evaluation_results.items():
            if 'aggregate' in results:
                r2 = results['aggregate']['mean_r2']
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model_name
        
        return {
            'n_models_evaluated': len(self.evaluation_results),
            'best_model': best_model,
            'best_r2': best_r2,
            'all_results': self.evaluation_results
        }
    
    def print_comparison(self, comparison_df: pd.DataFrame) -> None:
        """Print a formatted comparison table."""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(comparison_df.to_string(index=False))
        print("="*70)


def evaluate_weather_model(model: Any,
                          X_test: Union[np.ndarray, pd.DataFrame],
                          y_test: Union[np.ndarray, pd.DataFrame],
                          model_name: str = 'model') -> Dict[str, Any]:
    """
    Convenience function to evaluate a weather model.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test targets.
        model_name: Name of the model.
        
    Returns:
        Evaluation results.
    """
    evaluator = WeatherModelEvaluator()
    return evaluator.evaluate_model(model, X_test, y_test, model_name)


def compare_weather_models(models: Dict[str, Any],
                          X_test: Union[np.ndarray, pd.DataFrame],
                          y_test: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """
    Convenience function to compare multiple weather models.
    
    Args:
        models: Dictionary of model_name -> model.
        X_test: Test features.
        y_test: Test targets.
        
    Returns:
        Comparison DataFrame.
    """
    evaluator = WeatherModelEvaluator()
    comparison = evaluator.compare_models(models, X_test, y_test)
    evaluator.print_comparison(comparison)
    return comparison
