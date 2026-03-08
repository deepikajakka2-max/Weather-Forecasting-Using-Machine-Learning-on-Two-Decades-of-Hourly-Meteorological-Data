"""
AutoML Orchestrator for Weather Forecasting System.

Main class that coordinates all components of the AutoML pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import json
import logging

from .data_loader import WeatherDataLoader
from .preprocess import WeatherPreprocessor
from .eda import WeatherEDA
from .train import WeatherModelTrainer
from .evaluate import WeatherModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherAutoML:
    """
    Automated Machine Learning pipeline for weather forecasting.
    
    This class orchestrates all components:
    - Data loading and validation
    - Exploratory data analysis
    - Feature engineering and preprocessing
    - Multi-model training
    - Model evaluation and selection
    - Persistence and deployment
    """
    
    def __init__(self,
                 data_path: Optional[Union[str, Path]] = None,
                 output_dir: Union[str, Path] = './output',
                 forecast_horizons: List[int] = [1, 24],
                 target_variables: List[str] = ['temperature_c', 'precipitation_mm', 'relative_humidity_pct'],
                 random_state: int = 42):
        """
        Initialize the AutoML pipeline.
        
        Args:
            data_path: Path to the weather data CSV file.
            output_dir: Directory for saving outputs (models, reports, etc.).
            forecast_horizons: Hours ahead to forecast.
            target_variables: Weather variables to predict.
            random_state: Random seed for reproducibility.
        """
        self.data_path = Path(data_path) if data_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.forecast_horizons = forecast_horizons
        self.target_variables = target_variables
        self.random_state = random_state
        
        # Components
        self.loader: Optional[WeatherDataLoader] = None
        self.preprocessor: Optional[WeatherPreprocessor] = None
        self.eda: Optional[WeatherEDA] = None
        self.trainer: Optional[WeatherModelTrainer] = None
        self.evaluator: Optional[WeatherModelEvaluator] = None
        
        # Data states
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []
        self.target_names: List[str] = []
        
        # Results
        self.eda_report: Optional[Dict] = None
        self.training_results: Optional[Dict] = None
        self.evaluation_results: Optional[Dict] = None
        self.best_model_name: Optional[str] = None
        
        logger.info("WeatherAutoML initialized")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Forecast horizons: {self.forecast_horizons}")
        logger.info(f"  Target variables: {self.target_variables}")
    
    def load_data(self, data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Load weather data from CSV file.
        
        Args:
            data_path: Optional override for data path.
            
        Returns:
            Loaded DataFrame.
        """
        data_path = Path(data_path) if data_path else self.data_path
        if not data_path:
            raise ValueError("No data path specified")
        
        logger.info(f"Loading data from {data_path}")
        self.loader = WeatherDataLoader(data_path)
        self.raw_data = self.loader.load()
        
        logger.info(f"Loaded {len(self.raw_data):,} observations")
        logger.info(f"Columns: {list(self.raw_data.columns)}")
        
        return self.raw_data
    
    def run_eda(self, save_report: bool = True) -> Dict[str, Any]:
        """
        Run exploratory data analysis.
        
        Args:
            save_report: Whether to save the EDA report to disk.
            
        Returns:
            EDA report dictionary.
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Running exploratory data analysis...")
        self.eda = WeatherEDA(self.raw_data)
        self.eda.print_summary()
        self.eda_report = self.eda.generate_report()
        
        if save_report:
            report_path = self.output_dir / 'eda_report.json'
            with open(report_path, 'w') as f:
                json.dump(self.eda_report, f, indent=2, default=str)
            logger.info(f"EDA report saved to {report_path}")
        
        return self.eda_report
    
    def preprocess(self,
                  horizon: int = 1,
                  test_size: float = 0.2,
                  scale: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess data for model training.
        
        Args:
            horizon: Forecast horizon in hours.
            test_size: Fraction of data for testing.
            scale: Whether to scale features.
            
        Returns:
            Tuple of (X, y).
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info(f"Preprocessing data for {horizon}h forecast horizon...")
        
        self.preprocessor = WeatherPreprocessor(
            target_cols=self.target_variables
        )
        
        X, y, feature_names, target_names = self.preprocessor.prepare_for_training(
            self.raw_data, 
            horizon=horizon,
            scale=scale
        )
        
        self.feature_names = feature_names
        self.target_names = target_names
        
        # Split temporally
        split_idx = int(len(X) * (1 - test_size))
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]
        
        logger.info(f"Training set: {len(self.X_train):,} samples")
        logger.info(f"Test set: {len(self.X_test):,} samples")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Targets: {self.target_names}")
        
        return X, y
    
    def train(self,
             models: Optional[List[str]] = None,
             cross_validate: bool = False) -> Dict[str, Any]:
        """
        Train models on the preprocessed data.
        
        Args:
            models: List of model names to train. None for all available.
            cross_validate: Whether to perform cross-validation.
            
        Returns:
            Training results dictionary.
        """
        if self.X_train is None:
            raise ValueError("No preprocessed data. Call preprocess() first.")
        
        logger.info("Starting model training...")
        
        self.trainer = WeatherModelTrainer(
            models=models,
            random_state=self.random_state
        )
        
        self.training_results = self.trainer.train_all_models(
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )
        
        # Get best model
        self.best_model_name, _ = self.trainer.get_best_model()
        
        # Cross-validation for best model
        if cross_validate and self.best_model_name:
            logger.info(f"Cross-validating {self.best_model_name}...")
            cv_results = self.trainer.cross_validate(
                self.best_model_name,
                pd.concat([self.X_train, self.X_test]),
                pd.concat([self.y_train, self.y_test])
            )
            self.training_results[self.best_model_name]['cross_validation'] = cv_results
        
        # Print summary
        print(self.trainer.summary())
        
        return self.training_results
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate all trained models on test data.
        
        Returns:
            Evaluation results dictionary.
        """
        if self.trainer is None or not self.trainer.trained_models:
            raise ValueError("No models trained. Call train() first.")
        
        logger.info("Evaluating models...")
        
        self.evaluator = WeatherModelEvaluator()
        comparison = self.evaluator.compare_models(
            self.trainer.trained_models,
            self.X_test,
            self.y_test,
            self.target_names
        )
        
        self.evaluator.print_comparison(comparison)
        self.evaluation_results = self.evaluator.generate_report()
        
        return self.evaluation_results
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model.
        
        Returns:
            Tuple of (model_name, model_instance).
        """
        if self.trainer is None:
            raise ValueError("No models trained. Call train() first.")
        return self.trainer.get_best_model()
    
    def predict(self,
               X: pd.DataFrame,
               model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            X: Features to predict on.
            model_name: Model to use. None for best model.
            
        Returns:
            Predictions array.
        """
        if self.trainer is None:
            raise ValueError("No models trained. Call train() first.")
        
        model_name = model_name or self.best_model_name
        if model_name not in self.trainer.trained_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.trainer.trained_models[model_name]
        return model.predict(X.values if isinstance(X, pd.DataFrame) else X)
    
    def save_models(self, models: Optional[List[str]] = None) -> List[str]:
        """
        Save trained models to disk.
        
        Args:
            models: Models to save. None for all.
            
        Returns:
            List of saved file paths.
        """
        if self.trainer is None:
            raise ValueError("No models trained. Call train() first.")
        
        models = models or list(self.trainer.trained_models.keys())
        saved_paths = []
        
        models_dir = self.output_dir / 'models'
        for model_name in models:
            path = self.trainer.save_model(model_name, models_dir)
            saved_paths.append(path)
        
        return saved_paths
    
    def save_pipeline_config(self) -> str:
        """
        Save the pipeline configuration.
        
        Returns:
            Path to saved config file.
        """
        config = {
            'forecast_horizons': self.forecast_horizons,
            'target_variables': self.target_variables,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'best_model': self.best_model_name,
            'n_features': len(self.feature_names),
            'saved_at': datetime.now().isoformat()
        }
        
        config_path = self.output_dir / 'pipeline_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Pipeline config saved to {config_path}")
        return str(config_path)
    
    def run_full_pipeline(self,
                         data_path: Optional[Union[str, Path]] = None,
                         horizon: int = 1,
                         models: Optional[List[str]] = None,
                         save_outputs: bool = True) -> Dict[str, Any]:
        """
        Run the complete AutoML pipeline.
        
        Args:
            data_path: Path to weather data.
            horizon: Forecast horizon in hours.
            models: Models to train.
            save_outputs: Whether to save models and reports.
            
        Returns:
            Dictionary with all pipeline results.
        """
        print("\n" + "="*70)
        print("WEATHER AUTOML PIPELINE")
        print("="*70)
        start_time = datetime.now()
        
        # Step 1: Load data
        print("\n[Step 1/5] Loading data...")
        self.load_data(data_path)
        
        # Step 2: EDA
        print("\n[Step 2/5] Running exploratory data analysis...")
        self.run_eda(save_report=save_outputs)
        
        # Step 3: Preprocess
        print(f"\n[Step 3/5] Preprocessing for {horizon}h forecast...")
        self.preprocess(horizon=horizon)
        
        # Step 4: Train
        print("\n[Step 4/5] Training models...")
        self.train(models=models)
        
        # Step 5: Evaluate
        print("\n[Step 5/5] Evaluating models...")
        self.evaluate()
        
        # Save outputs
        if save_outputs:
            print("\nSaving outputs...")
            self.save_models()
            self.save_pipeline_config()
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Best model: {self.best_model_name}")
        if self.evaluation_results and 'best_r2' in self.evaluation_results:
            print(f"  Best R² score: {self.evaluation_results['best_r2']:.4f}")
        print(f"  Outputs saved to: {self.output_dir}")
        print("="*70 + "\n")
        
        return {
            'eda': self.eda_report,
            'training': self.training_results,
            'evaluation': self.evaluation_results,
            'best_model': self.best_model_name,
            'total_time_seconds': total_time
        }
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from the best model."""
        if self.trainer:
            return self.trainer.get_feature_importance()
        return None
    
    def summary(self) -> None:
        """Print pipeline summary."""
        print("\n" + "="*70)
        print("WEATHER AUTOML PIPELINE SUMMARY")
        print("="*70)
        
        if self.raw_data is not None:
            print(f"\nData: {len(self.raw_data):,} observations")
            if self.loader:
                info = self.loader.get_station_info()
                print(f"Station: {info.get('name', 'Unknown')}")
        
        if self.X_train is not None:
            print(f"\nFeatures: {len(self.feature_names)}")
            print(f"Targets: {self.target_names}")
            print(f"Train size: {len(self.X_train):,}")
            print(f"Test size: {len(self.X_test):,}")
        
        if self.best_model_name:
            print(f"\nBest Model: {self.best_model_name}")
            if self.evaluation_results:
                print(f"Best R²: {self.evaluation_results.get('best_r2', 'N/A'):.4f}")
        
        print("="*70 + "\n")


def create_weather_pipeline(data_path: Union[str, Path],
                           output_dir: str = './output',
                           **kwargs) -> WeatherAutoML:
    """
    Factory function to create a weather AutoML pipeline.
    
    Args:
        data_path: Path to weather data.
        output_dir: Output directory.
        **kwargs: Additional arguments for WeatherAutoML.
        
    Returns:
        Configured WeatherAutoML instance.
    """
    return WeatherAutoML(
        data_path=data_path,
        output_dir=output_dir,
        **kwargs
    )
