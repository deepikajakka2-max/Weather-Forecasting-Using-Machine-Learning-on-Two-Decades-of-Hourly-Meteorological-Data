"""
Weather Forecasting AutoML Package.

An automated machine learning system for weather prediction using
historical meteorological data.
"""

from .data_loader import WeatherDataLoader, load_weather_data
from .preprocess import WeatherPreprocessor, preprocess_weather_data
from .eda import WeatherEDA, run_eda
from .train import WeatherModelTrainer, train_weather_models
from .evaluate import WeatherModelEvaluator, evaluate_weather_model, compare_weather_models
from .automl import WeatherAutoML, create_weather_pipeline

__version__ = '0.1.0'
__author__ = 'RBY-Team'

__all__ = [
    # Main orchestrator
    'WeatherAutoML',
    'create_weather_pipeline',
    
    # Data loading
    'WeatherDataLoader',
    'load_weather_data',
    
    # Preprocessing
    'WeatherPreprocessor',
    'preprocess_weather_data',
    
    # EDA
    'WeatherEDA',
    'run_eda',
    
    # Training
    'WeatherModelTrainer',
    'train_weather_models',
    
    # Evaluation
    'WeatherModelEvaluator',
    'evaluate_weather_model',
    'compare_weather_models',
]
