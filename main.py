"""
Main orchestration script for the Crop Yield Prediction project.
Runs the complete ML pipeline: loading data, EDA, preprocessing, training, evaluation, and prediction.
"""

import sys
import argparse
from pathlib import Path

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from data_loader import load_data, validate_required_columns, get_data_info
from eda import perform_eda
from preprocessing import build_preprocessor
from train import split_data, train_baseline_model, tune_model, perform_cross_validation
from evaluate import evaluate_model, print_evaluation_report
from predict import (get_feature_importance, plot_feature_importance, 
                     save_feature_importance_csv, print_top_features,
                     predict_crop_yield)
from utils import create_directories, save_dict_to_json, load_dict_from_json, save_model, load_model


# Configuration
CONFIG = {
    'data_path': 'data/crop_yield_data.csv',
    'models_dir': 'models',
    'outputs_dir': 'outputs',
    'model_save_path': 'models/crop_yield_rf_pipeline.joblib',
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'required_columns': ['N', 'P', 'K', 'pH', 'organic_matter', 'rainfall', 
                        'temp_min', 'temp_max', 'fertilizer_usage', 'crop_type', 'crop_yield'],
    'target_column': 'crop_yield',
    'categorical_column': 'crop_type'
}


def main():
    """
    Execute the complete ML pipeline.
    """
    
    parser = argparse.ArgumentParser(
        description="Crop Yield Prediction pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "eda-only", "train-only", "predict-only"],
        default="full",
        help="Execution mode: full pipeline, EDA only, training only, or prediction only"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive prediction mode after training"
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="Path to JSON input file for --mode predict-only"
    )
    args = parser.parse_args()

    print("\n" + "="*70)
    print("CROP YIELD PREDICTION - MACHINE LEARNING PIPELINE")
    print("="*70 + "\n")
    
    try:
        # Step 1: Create necessary directories
        print("STEP 1: Creating output directories...")
        create_directories([CONFIG['models_dir'], CONFIG['outputs_dir']])
        print(f"✓ Created: {CONFIG['models_dir']}, {CONFIG['outputs_dir']}\n")

        if args.mode == "predict-only":
            print("STEP 2: Loading trained model for prediction...")
            model = load_model(CONFIG['model_save_path'])
            print(f"✓ Model loaded from: {CONFIG['model_save_path']}\n")

            if args.input_json:
                sample_input = load_dict_from_json(args.input_json)
                print(f"Loaded prediction input from: {args.input_json}")
            else:
                sample_input = {
                    "N": 90,
                    "P": 40,
                    "K": 40,
                    "pH": 6.5,
                    "organic_matter": 2.8,
                    "rainfall": 220,
                    "temp_min": 18,
                    "temp_max": 31,
                    "fertilizer_usage": 120,
                    "crop_type": "rice"
                }
                print("Using default sample input (provide --input-json to override)")

            prediction = predict_crop_yield(model, sample_input, feature_names=[])
            print("\n" + "="*70)
            print("PREDICTION RESULT")
            print("="*70)
            print(f"Input: {sample_input}")
            print(f"Predicted Crop Yield: {prediction:.2f} units")
            print("="*70 + "\n")
            return
        
        # Step 2: Load data
        print("STEP 2: Loading data...")
        df = load_data(CONFIG['data_path'])
        print(f"✓ Data loaded successfully from: {CONFIG['data_path']}")
        print(f"  Shape: {df.shape}")
        validate_required_columns(df, CONFIG['required_columns'])
        print(f"✓ All required columns present\n")
        
        # Show data info
        data_info = get_data_info(df)
        print(f"Data Info:")
        print(f"  Memory Usage: {data_info['memory_usage']:.2f} MB\n")

        if args.mode in ["full", "eda-only"]:
            # Step 3: Exploratory Data Analysis (EDA)
            print("STEP 3: Performing Exploratory Data Analysis...")
            perform_eda(df, CONFIG['target_column'], CONFIG['categorical_column'], 
                       CONFIG['outputs_dir'])
            print("✓ EDA completed and plots saved\n")

            if args.mode == "eda-only":
                print("="*70)
                print("EDA-ONLY EXECUTION COMPLETED SUCCESSFULLY")
                print("="*70 + "\n")
                return
        
        # Step 4: Data preprocessing
        print("STEP 4: Building data preprocessor...")
        preprocessor, feature_names = build_preprocessor(df, CONFIG['target_column'],
                                                        CONFIG['categorical_column'])
        print(f"✓ Preprocessor built successfully")
        print(f"  Number of features after transformation: {len(feature_names)}")
        print(f"  Features: {feature_names[:5]}... (showing first 5)\n")
        
        # Step 5: Train-test split
        print("STEP 5: Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = split_data(df, CONFIG['target_column'],
                                                        CONFIG['test_size'],
                                                        CONFIG['random_state'])
        print(f"✓ Data split completed")
        print(f"  Train set: {X_train.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples\n")
        
        # Step 6: Train baseline model
        print("STEP 6: Training baseline Random Forest model...")
        baseline_pipeline = train_baseline_model(X_train, y_train, preprocessor,
                                                 CONFIG['random_state'])
        print("✓ Baseline model trained\n")
        
        # Step 7: Hyperparameter tuning
        print("STEP 7: Tuning hyperparameters with GridSearchCV...")
        best_pipeline, tune_results = tune_model(X_train, y_train, preprocessor,
                                                  CONFIG['cv_folds'],
                                                  CONFIG['random_state'])
        print("✓ Hyperparameter tuning completed\n")
        
        # Step 8: Cross-validation
        print("STEP 8: Performing cross-validation on best model...")
        cv_results = perform_cross_validation(best_pipeline, X_train, y_train,
                                             CONFIG['cv_folds'])
        print(f"✓ Cross-validation completed")
        print(f"  Mean CV R² Score: {cv_results['r2_mean']:.4f} (+/- {cv_results['r2_std']:.4f})")
        print(f"  Mean CV RMSE: {cv_results['rmse_mean']:.4f} (+/- {cv_results['rmse_std']:.4f})\n")
        
        # Step 9: Model evaluation
        print("STEP 9: Evaluating model on test set...")
        eval_results = evaluate_model(best_pipeline, X_train, X_test, y_train, y_test)
        eval_summary = print_evaluation_report(eval_results)
        
        # Step 10: Feature importance
        print("STEP 10: Extracting feature importance...")
        importance_df = get_feature_importance(best_pipeline, feature_names)
        save_feature_importance_csv(importance_df, CONFIG['outputs_dir'])
        plot_feature_importance(importance_df, top_n=15, output_dir=CONFIG['outputs_dir'])
        print_top_features(importance_df, top_n=10)
        
        # Step 11: Save model
        print("STEP 11: Saving trained model...")
        save_model(best_pipeline, CONFIG['model_save_path'])
        print(f"✓ Model saved to: {CONFIG['model_save_path']}\n")
        
        # Step 12: Test prediction function
        print("STEP 12: Testing prediction function...")
        sample_input = {
            "N": 90,
            "P": 40,
            "K": 40,
            "pH": 6.5,
            "organic_matter": 2.8,
            "rainfall": 220,
            "temp_min": 18,
            "temp_max": 31,
            "fertilizer_usage": 120,
            "crop_type": "rice"
        }
        
        predicted_yield = predict_crop_yield(best_pipeline, sample_input, feature_names)
        print(f"✓ Prediction successful!")
        print(f"  Sample Input: {sample_input}")
        print(f"  Predicted Crop Yield: {predicted_yield:.2f} units\n")
        
        # Step 13: Interactive prediction mode
        if args.interactive and args.mode == "full":
            print("STEP 13: Starting interactive prediction mode...")
            interactive_predict_mode(best_pipeline, feature_names)
        else:
            print("STEP 13: Interactive prediction mode skipped (use --interactive to enable)")
        
        # Save configuration and results summary
        results_summary = {
            'model_path': CONFIG['model_save_path'],
            'data_shape': df.shape,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': feature_names,
            'best_hyperparameters': tune_results['best_params'],
            'cv_r2_score': float(cv_results['r2_mean']),
            'evaluation': eval_summary
        }
        
        summary_path = f"{CONFIG['outputs_dir']}/results_summary.json"
        save_dict_to_json(results_summary, summary_path)
        print(f"\n✓ Results summary saved to: {summary_path}")
        
        print("\n" + "="*70)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def interactive_predict_mode(model, feature_names):
    """
    Interactive mode for making predictions.
    User can input values or skip to exit.
    
    Parameters
    ----------
    model : Pipeline
        Trained model pipeline
    feature_names : list
        List of feature names
    """
    
    print("="*70)
    print("INTERACTIVE PREDICTION MODE")
    print("="*70)
    print("Enter crop details to predict yield. Type 'exit' or 'quit' to stop.\n")
    
    crop_types = ['rice', 'maize', 'wheat', 'barley', 'oats']
    
    while True:
        try:
            print("\nEnter the following values:")
            user_input = {}
            
            # Get numerical inputs
            for col in ['N', 'P', 'K']:
                val = input(f"{col} (kg/ha) [20-100]: ").strip()
                if val.lower() in ['exit', 'quit']:
                    return
                user_input[col] = float(val)
            
            user_input['pH'] = float(input("pH level [5.5-8.5]: ").strip())
            user_input['organic_matter'] = float(input("Organic Matter (%) [1-5]: ").strip())
            user_input['rainfall'] = float(input("Rainfall (mm) [100-300]: ").strip())
            user_input['temp_min'] = float(input("Min Temperature (°C) [10-25]: ").strip())
            user_input['temp_max'] = float(input("Max Temperature (°C) [25-40]: ").strip())
            user_input['fertilizer_usage'] = float(input("Fertilizer Usage (kg/ha) [50-200]: ").strip())
            
            print(f"\nCrop Type options: {', '.join(crop_types)}")
            crop = input("Select crop type: ").strip().lower()
            if crop not in crop_types:
                print(f"Invalid crop type. Please choose from: {crop_types}")
                continue
            user_input['crop_type'] = crop
            
            # Make prediction
            try:
                prediction = predict_crop_yield(model, user_input, feature_names)
                print(f"\n{'='*50}")
                print(f"Predicted Crop Yield: {prediction:.2f} units")
                print(f"{'='*50}\n")
            except ValueError as e:
                print(f"Prediction error: {e}\n")
                
        except ValueError:
            print("Invalid input. Please enter numeric values.\n")
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            return


if __name__ == '__main__':
    main()
