"""
================================================================================
MODEL TRAINING MODULE - Weather Forecasting AutoML System
================================================================================

PURPOSE:
    Train multiple machine learning models for weather forecasting.
    Compare algorithms to find the best one for your data.

AVAILABLE MODELS:
    - Ridge Regression: Linear model with L2 regularization (fast, interpretable)
    - Lasso Regression: Linear model with L1 regularization (feature selection)
    - ElasticNet: Combines L1 and L2 regularization
    - Decision Tree: Non-linear, interpretable rules
    - Random Forest: Ensemble of trees (usually best accuracy)
    - Gradient Boosting: Sequential ensemble (very accurate)
    - KNN: K-Nearest Neighbors (simple, no assumptions)
    - SVR: Support Vector Regression (good for complex patterns)

USAGE:
    from automl.train import WeatherModelTrainer
    
    trainer = WeatherModelTrainer(models=['ridge', 'random_forest'])
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    best_name, best_model = trainer.get_best_model()

AUTHOR: Deepika Jakka
DATE: March 2026
================================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd      # DataFrame handling
import numpy as np       # Numerical operations
from typing import Dict, Any, List, Optional, Tuple, Union  # Type hints
from pathlib import Path  # Path handling
import pickle            # Model serialization
import json              # JSON for metadata
from datetime import datetime  # Timestamps
import logging           # Status logging

# Scikit-learn imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor  # Wrapper for multi-target

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# WEATHER MODEL TRAINER CLASS
# =============================================================================
class WeatherModelTrainer:
    """
    Train and manage weather forecasting models.
    
    This class handles:
    - Creating model instances with default hyperparameters
    - Training single or multiple models
    - Time series cross-validation
    - Model saving and loading
    - Feature importance extraction
    
    Attributes:
        models_to_train: List of model names to train
        random_state: Random seed for reproducibility
        trained_models: Dictionary of trained model instances
        training_results: Dictionary of training metrics
        best_model_name: Name of the best performing model
        feature_names: List of feature column names
        target_names: List of target column names
    """
    
    # -------------------------------------------------------------------------
    # MODEL CONFIGURATIONS
    # -------------------------------------------------------------------------
    # Each model has: class, default parameters, and multioutput support flag
    MODELS = {
        'ridge': {
            'class': Ridge,  # Linear regression with L2 penalty
            'params': {'alpha': 1.0},  # Regularization strength
            'supports_multioutput': True  # Can predict multiple targets
        },
        'lasso': {
            'class': Lasso,  # Linear regression with L1 penalty (sparse)
            'params': {'alpha': 0.1, 'max_iter': 2000},
            'supports_multioutput': False  # Needs MultiOutputRegressor wrapper
        },
        'elastic_net': {
            'class': ElasticNet,  # Combines L1 and L2
            'params': {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 2000},
            'supports_multioutput': False
        },
        'decision_tree': {
            'class': DecisionTreeRegressor,  # Tree-based, interpretable
            'params': {'max_depth': 15, 'min_samples_split': 10},
            'supports_multioutput': True
        },
        'random_forest': {
            'class': RandomForestRegressor,  # Ensemble of trees
            'params': {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 10, 'n_jobs': -1},
            'supports_multioutput': True
        },
        'gradient_boosting': {
            'class': GradientBoostingRegressor,  # Sequential boosting
            'params': {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1},
            'supports_multioutput': False
        },
        'knn': {
            'class': KNeighborsRegressor,  # Instance-based learning
            'params': {'n_neighbors': 5, 'weights': 'distance'},
            'supports_multioutput': True
        },
        'svr': {
            'class': SVR,  # Support Vector Regression
            'params': {'kernel': 'rbf', 'C': 1.0},
            'supports_multioutput': False
        }
    }
    
    # -------------------------------------------------------------------------
    # HYPERPARAMETER SEARCH SPACES (for future tuning)
    # -------------------------------------------------------------------------
    PARAM_GRIDS = {
        'ridge': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
        'lasso': {'alpha': [0.001, 0.01, 0.1, 1.0]},
        'elastic_net': {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]},
        'decision_tree': {'max_depth': [5, 10, 15, 20], 'min_samples_split': [5, 10, 20]},
        'random_forest': {'n_estimators': [50, 100, 200], 'max_depth': [10, 15, 20]},
        'gradient_boosting': {'n_estimators': [50, 100], 'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1, 0.2]},
        'knn': {'n_neighbors': [3, 5, 10, 15]},
        'svr': {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear']}
    }
    
    # -------------------------------------------------------------------------
    # CONSTRUCTOR
    # -------------------------------------------------------------------------
    def __init__(self, 
                 models: Optional[List[str]] = None,
                 random_state: int = 42):
        """
        Initialize the trainer.
        
        Args:
            models: List of model names to train. If None, trains all available.
                   Options: 'ridge', 'lasso', 'elastic_net', 'decision_tree',
                           'random_forest', 'gradient_boosting', 'knn', 'svr'
            random_state: Random seed for reproducibility (default: 42)
            
        Example:
            # Train all models
            trainer = WeatherModelTrainer()
            
            # Train specific models only
            trainer = WeatherModelTrainer(models=['ridge', 'random_forest'])
        """
        # Use all available models if none specified
        self.models_to_train = models or list(self.MODELS.keys())
        
        # Random seed for reproducible results
        self.random_state = random_state
        
        # Storage for trained models and results
        self.trained_models: Dict[str, Any] = {}
        self.training_results: Dict[str, Dict] = {}
        self.best_model_name: Optional[str] = None
        
        # Feature and target column names (populated during training)
        self.feature_names: List[str] = []
        self.target_names: List[str] = []
        
    # -------------------------------------------------------------------------
    # CREATE MODEL INSTANCE
    # -------------------------------------------------------------------------
    def _create_model(self, 
                     model_name: str, 
                     custom_params: Optional[Dict] = None,
                     multioutput: bool = False) -> Any:
        """
        Create a model instance with specified parameters.
        
        Args:
            model_name: Name of the model (e.g., 'ridge', 'random_forest')
            custom_params: Custom parameters to override defaults
            multioutput: If True, wrap in MultiOutputRegressor if needed
            
        Returns:
            Configured model instance ready for training
            
        Raises:
            ValueError: If model_name is not recognized
        """
        # Validate model name
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")
        
        # Get model configuration
        model_config = self.MODELS[model_name]
        params = model_config['params'].copy()  # Copy to avoid modifying original
        
        # Override with custom parameters if provided
        if custom_params:
            params.update(custom_params)
        
        # Add random_state if the model supports it
        if 'random_state' in model_config['class'].__init__.__code__.co_varnames:
            params['random_state'] = self.random_state
        
        # Create model instance
        model = model_config['class'](**params)
        
        # Wrap in MultiOutputRegressor if needed for multi-target prediction
        if multioutput and not model_config['supports_multioutput']:
            model = MultiOutputRegressor(model, n_jobs=-1)
        
        return model
    
    # -------------------------------------------------------------------------
    # TEMPORAL TRAIN/TEST SPLIT
    # -------------------------------------------------------------------------
    def train_test_split_temporal(self,
                                  X: pd.DataFrame,
                                  y: pd.DataFrame,
                                  test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data preserving temporal order (no shuffling).
        
        IMPORTANT: For time series data, we MUST NOT shuffle!
        - Training data should be EARLIER in time
        - Test data should be LATER in time
        - This simulates real forecasting (predict future from past)
        
        Args:
            X: Features DataFrame
            y: Target DataFrame
            test_size: Fraction of data for testing (default: 0.2 = 20%)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Calculate split index (test data is the last portion)
        split_idx = int(len(X) * (1 - test_size))
        
        # Split by index (preserves time order)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    # -------------------------------------------------------------------------
    # TRAIN SINGLE MODEL
    # -------------------------------------------------------------------------
    def train_single_model(self,
                          model_name: str,
                          X_train: pd.DataFrame,
                          y_train: pd.DataFrame,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train a single model and return results.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional, for validation score)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary with training results:
            - model_name: Name of the model
            - training_time_seconds: How long training took
            - train_r2: R² score on training data
            - val_r2: R² score on validation data (if provided)
            - n_features: Number of features
            - n_samples: Number of training samples
        """
        logger.info(f"Training {model_name}...")
        start_time = datetime.now()
        
        # Check if we have multiple targets
        multioutput = y_train.shape[1] > 1 if len(y_train.shape) > 1 else False
        
        # Create model instance
        model = self._create_model(model_name, multioutput=multioutput)
        
        # Convert to numpy for training (sklearn prefers numpy)
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.DataFrame) else y_train
        
        # Flatten y if single target
        if y_train_np.ndim == 2 and y_train_np.shape[1] == 1:
            y_train_np = y_train_np.ravel()
        
        # Train the model
        model.fit(X_train_np, y_train_np)
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate training R² score
        train_score = model.score(X_train_np, y_train_np)
        
        # Build results dictionary
        result = {
            'model_name': model_name,
            'training_time_seconds': training_time,
            'train_r2': train_score,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }
        
        # Calculate validation score if validation data provided
        if X_val is not None and y_val is not None:
            X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_np = y_val.values if isinstance(y_val, pd.DataFrame) else y_val
            if y_val_np.ndim == 2 and y_val_np.shape[1] == 1:
                y_val_np = y_val_np.ravel()
            result['val_r2'] = model.score(X_val_np, y_val_np)
        
        # Store trained model and results
        self.trained_models[model_name] = model
        self.training_results[model_name] = result
        
        logger.info(f"  {model_name}: R² = {train_score:.4f} (train), time = {training_time:.2f}s")
        
        return result
    
    # -------------------------------------------------------------------------
    # TRAIN ALL MODELS
    # -------------------------------------------------------------------------
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.DataFrame,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[pd.DataFrame] = None) -> Dict[str, Dict]:
        """
        Train all specified models and compare performance.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional but recommended)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary mapping model_name -> training_results
        """
        # Store feature and target names for later use
        self.feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else []
        self.target_names = list(y_train.columns) if isinstance(y_train, pd.DataFrame) else []
        
        results = {}
        
        # Train each model
        for model_name in self.models_to_train:
            try:
                results[model_name] = self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val
                )
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Determine best model (by validation R² if available, else training R²)
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            score_key = 'val_r2' if X_val is not None else 'train_r2'
            self.best_model_name = max(
                valid_results, 
                key=lambda x: valid_results[x].get(score_key, -np.inf)
            )
            logger.info(f"\nBest model: {self.best_model_name}")
        
        return results
    
    # -------------------------------------------------------------------------
    # TIME SERIES CROSS-VALIDATION
    # -------------------------------------------------------------------------
    def cross_validate(self,
                      model_name: str,
                      X: pd.DataFrame,
                      y: pd.DataFrame,
                      n_splits: int = 5) -> Dict[str, Any]:
        """
        Perform time series cross-validation.
        
        TimeSeriesSplit creates folds that respect temporal ordering:
        - Fold 1: train on first 1/n, test on next
        - Fold 2: train on first 2/n, test on next
        - etc.
        
        Args:
            model_name: Model to cross-validate
            X: Features
            y: Targets
            n_splits: Number of CV folds (default: 5)
            
        Returns:
            Dictionary with CV scores and statistics
        """
        logger.info(f"Cross-validating {model_name} with {n_splits} splits...")
        
        # Use TimeSeriesSplit (respects time order)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Check for multi-output
        multioutput = y.shape[1] > 1 if len(y.shape) > 1 else False
        
        # Create fresh model instance
        model = self._create_model(model_name, multioutput=multioutput)
        
        # Convert to numpy
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.DataFrame) else y
        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np.ravel()
        
        # Run cross-validation
        scores = cross_val_score(model, X_np, y_np, cv=tscv, scoring='r2')
        
        return {
            'model_name': model_name,
            'cv_scores': scores.tolist(),
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std(),
            'n_splits': n_splits
        }
    
    # -------------------------------------------------------------------------
    # GET BEST MODEL
    # -------------------------------------------------------------------------
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model.
        
        Returns:
            Tuple of (model_name, model_instance)
            
        Raises:
            ValueError: If no models have been trained yet
        """
        if self.best_model_name is None:
            raise ValueError("No models trained yet. Call train_all_models() first.")
        return self.best_model_name, self.trained_models[self.best_model_name]
    
    # -------------------------------------------------------------------------
    # GET FEATURE IMPORTANCE
    # -------------------------------------------------------------------------
    def get_feature_importance(self, model_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get feature importance for tree-based models or coefficients for linear models.
        
        Args:
            model_name: Model to get importance for. Uses best model if None.
            
        Returns:
            DataFrame with feature importances, sorted by importance, or None if not supported
        """
        model_name = model_name or self.best_model_name
        if model_name is None:
            return None
        
        model = self.trained_models.get(model_name)
        if model is None:
            return None
        
        # Handle MultiOutputRegressor (use first estimator)
        if isinstance(model, MultiOutputRegressor):
            model = model.estimators_[0]
        
        # Tree-based models have feature_importances_
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        
        # Linear models have coefficients
        elif hasattr(model, 'coef_'):
            coef = model.coef_ if model.coef_.ndim == 1 else model.coef_[0]
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coef,
                'abs_coefficient': np.abs(coef)
            }).sort_values('abs_coefficient', ascending=False)
            return importance
        
        return None
    
    # -------------------------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------------------------
    def save_model(self, 
                  model_name: str,
                  save_path: Union[str, Path]) -> str:
        """
        Save a trained model to disk as a pickle file.
        
        Args:
            model_name: Name of the model to save
            save_path: Directory to save the model
            
        Returns:
            Path to the saved model file
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.trained_models.keys())}")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model as pickle
        model_file = save_path / f"{model_name}_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self.trained_models[model_name], f)
        
        # Save metadata as JSON
        metadata = {
            'model_name': model_name,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'training_results': self.training_results.get(model_name, {}),
            'saved_at': datetime.now().isoformat()
        }
        
        meta_file = save_path / f"{model_name}_metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved {model_name} to {model_file}")
        return str(model_file)
    
    # -------------------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------------------
    def load_model(self, model_path: Union[str, Path]) -> Any:
        """
        Load a saved model from disk.
        
        Args:
            model_path: Path to the model pickle file
            
        Returns:
            Loaded model instance
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Try to load metadata
        meta_path = Path(model_path).with_suffix('.json').with_name(
            Path(model_path).stem.replace('_model', '_metadata') + '.json'
        )
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
                self.target_names = metadata.get('target_names', [])
        
        return model
    
    # -------------------------------------------------------------------------
    # GET TRAINING SUMMARY
    # -------------------------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """
        Get a summary of all training results as a DataFrame.
        
        Returns:
            DataFrame with columns: model, status, train_r2, val_r2, training_time
        """
        if not self.training_results:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, results in self.training_results.items():
            if 'error' in results:
                summary_data.append({
                    'model': model_name,
                    'status': 'error',
                    'train_r2': None,
                    'val_r2': None,
                    'training_time': None
                })
            else:
                summary_data.append({
                    'model': model_name,
                    'status': 'success',
                    'train_r2': results.get('train_r2'),
                    'val_r2': results.get('val_r2'),
                    'training_time': results.get('training_time_seconds')
                })
        
        # Sort by validation R² (or training R²)
        sort_col = 'val_r2' if 'val_r2' in summary_data[0] else 'train_r2'
        return pd.DataFrame(summary_data).sort_values(sort_col, ascending=False, na_position='last')


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================
def train_weather_models(X: pd.DataFrame, 
                        y: pd.DataFrame,
                        test_size: float = 0.2,
                        models: Optional[List[str]] = None) -> Tuple[WeatherModelTrainer, Dict]:
    """
    Convenience function to train weather forecasting models in one call.
    
    Args:
        X: Features DataFrame
        y: Target DataFrame
        test_size: Fraction for test split (default: 0.2)
        models: Models to train (None for all available)
        
    Returns:
        Tuple of (trainer_instance, results_dictionary)
        
    Example:
        trainer, results = train_weather_models(X, y, models=['ridge', 'random_forest'])
        best_name, best_model = trainer.get_best_model()
    """
    trainer = WeatherModelTrainer(models=models)
    X_train, X_test, y_train, y_test = trainer.train_test_split_temporal(X, y, test_size)
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    return trainer, results
