#!/usr/bin/env python3
"""
Weather Forecasting AutoML System - Command Line Interface

Usage:
    python main.py                    # Run full pipeline with defaults
    python main.py --data path/to.csv # Specify data file
    python main.py --horizon 24       # 24-hour forecast
    python main.py --models rf ridge  # Train specific models only
"""

import argparse
import sys
from pathlib import Path

from automl import WeatherAutoML, create_weather_pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Weather Forecasting AutoML System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run with defaults
  python main.py --data hly175.csv         # Use specific data file
  python main.py --horizon 24              # 24-hour forecast
  python main.py --models random_forest ridge  # Train specific models
  python main.py --eda-only                # Only run EDA
        """
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='hly175.csv',
        help='Path to weather data CSV file (default: hly175.csv)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./output',
        help='Output directory for models and reports (default: ./output)'
    )
    
    parser.add_argument(
        '--horizon', '-H',
        type=int,
        default=1,
        choices=[1, 6, 12, 24],
        help='Forecast horizon in hours (default: 1)'
    )
    
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        choices=['ridge', 'lasso', 'elastic_net', 'decision_tree', 
                 'random_forest', 'gradient_boosting', 'knn', 'svr'],
        help='Models to train (default: all)'
    )
    
    parser.add_argument(
        '--targets', '-t',
        nargs='+',
        default=['temperature_c', 'precipitation_mm', 'relative_humidity_pct'],
        help='Target variables to predict'
    )
    
    parser.add_argument(
        '--eda-only',
        action='store_true',
        help='Only run exploratory data analysis'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save models and reports'
    )
    
    parser.add_argument(
        '--cross-validate',
        action='store_true',
        help='Perform cross-validation on best model'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the Weather AutoML system."""
    args = parse_args()
    
    # Validate data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("WEATHER FORECASTING AUTOML SYSTEM")
    print("RBY-Team - Phoenix Park Weather Prediction")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Data file: {data_path}")
    print(f"  Output directory: {args.output}")
    print(f"  Forecast horizon: {args.horizon}h")
    print(f"  Models: {args.models or 'all'}")
    print(f"  Targets: {args.targets}")
    
    # Create pipeline
    pipeline = WeatherAutoML(
        data_path=data_path,
        output_dir=args.output,
        forecast_horizons=[args.horizon],
        target_variables=args.targets,
        random_state=args.seed
    )
    
    # EDA only mode
    if args.eda_only:
        print("\n[EDA-only mode]")
        pipeline.load_data()
        pipeline.run_eda(save_report=not args.no_save)
        print("\nEDA complete.")
        return
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        horizon=args.horizon,
        models=args.models,
        save_outputs=not args.no_save
    )
    
    # Cross-validation
    if args.cross_validate and pipeline.best_model_name:
        print("\nRunning cross-validation...")
        cv_results = pipeline.trainer.cross_validate(
            pipeline.best_model_name,
            pipeline.X_train,
            pipeline.y_train
        )
        print(f"CV Mean R²: {cv_results['mean_cv_score']:.4f} (+/- {cv_results['std_cv_score']:.4f})")
    
    # Show feature importance
    importance = pipeline.get_feature_importance()
    if importance is not None:
        print("\nTop 10 Most Important Features:")
        print(importance.head(10).to_string(index=False))
    
    # Summary
    pipeline.summary()
    
    print("\nDone! Check the output directory for saved models and reports.")
    return results


if __name__ == '__main__':
    main()
