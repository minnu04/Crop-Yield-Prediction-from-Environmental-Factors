# Crop Yield Prediction from Environmental Factors

A complete, production-grade machine learning pipeline for predicting agricultural crop yield based on environmental and agronomic factors.

## 📋 Project Overview

This project builds an end-to-end machine learning solution that predicts crop yield using historical agricultural data. The model analyzes relationships between environmental variables (temperature, rainfall, pH) and agronomic inputs (fertilizers, nutrients) to forecast yield with high accuracy.

**Status:** ✅ Ready for production use and academic submission

## 🎯 Problem Statement

Modern agriculture faces the challenge of optimizing yield given variable environmental conditions and resource constraints. Accurate yield prediction enables:
- Better resource allocation (fertilizers, water)
- Risk assessment and planning
- Precision agriculture decision-making
- Farm management optimization

This project demonstrates how machine learning can address this real-world problem using Random Forest regression with comprehensive data analysis and hyperparameter tuning.

## 📊 Dataset Schema

The model expects a CSV file (`crop_yield_data.csv`) with the following columns:

| Column Name | Data Type | Range | Description |
|---|---|---|---|
| N | Float | 20-100 | Nitrogen content (kg/ha) |
| P | Float | 5-80 | Phosphorus content (kg/ha) |
| K | Float | 5-80 | Potassium content (kg/ha) |
| pH | Float | 5.5-8.5 | Soil pH level |
| organic_matter | Float | 1-5 | Organic matter percentage (%) |
| rainfall | Float | 100-300 | Annual rainfall (mm) |
| temp_min | Float | 10-25 | Minimum temperature (°C) |
| temp_max | Float | 25-40 | Maximum temperature (°C) |
| fertilizer_usage | Float | 50-200 | Total fertilizer applied (kg/ha) |
| crop_type | String | {rice, maize, wheat, barley, oats} | Type of crop |
| crop_yield | Float | ≥5 | **Target variable** - crop yield (units) |

**Note:** The project is robust to extra unused columns in the CSV.

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project
```bash
cd crop-yield-prediction/
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 How to Run

### Quick Start

1. **Generate Sample Data** (optional - for testing):
   ```bash
   python generate_sample_data.py
   ```
   This creates `data/crop_yield_data.csv` with 1000 synthetic samples.

2. **Run Full ML Pipeline**:
   ```bash
   python main.py
   ```

   The full pipeline will:
   - Load and validate data
   - Perform exploratory data analysis (EDA)
   - Build and fit preprocessor
   - Split data into train/test sets
   - Train baseline Random Forest model
   - Tune hyperparameters using GridSearchCV
   - Evaluate on test set
   - Extract and visualize feature importance
   - Save the trained model

### CLI Modes

Use quick execution modes with `--mode`:

```bash
# Run only EDA and generate plots
python main.py --mode eda-only

# Run training/tuning/evaluation and save model (skips EDA)
python main.py --mode train-only

# Run one-off prediction using saved model and default sample input
python main.py --mode predict-only

# Run one-off prediction using custom JSON input
python main.py --mode predict-only --input-json data/sample_input.json

# Run full pipeline + interactive prediction prompt
python main.py --mode full --interactive
```

### Run Tests

```bash
python -m unittest discover -s tests -v
```

### Web App (Role-Based Access)

The project also includes a website with role-based access for:
- admin: user management, system usage dashboard, prediction monitoring
- farmer: run crop yield predictions and view personal prediction history

Run the web app:

```bash
python web/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Default admin credentials:

```text
username: admin
password: admin123
```

Important:
- change the default admin password after first login by creating a new admin user and removing the default account.
- if model file is missing, train first using `python main.py --mode train-only`.

### Weather-Assisted Mode Configuration

Optional environment variables for weather integration:

```text
WEATHER_PROVIDER=open-meteo
WEATHER_API_KEY=your_openweather_key_if_using_openweather
WEATHER_TIMEOUT_SECONDS=10
```

Recommended setup:

1. Copy `.env.example` to `.env` in the project root.
2. For free weather access with no key, keep `WEATHER_PROVIDER=open-meteo`.
3. If you want to provide your own API key, set:

```text
WEATHER_PROVIDER=openweather
WEATHER_API_KEY=your_actual_openweather_api_key
```

The app now loads `.env` automatically at startup.

Notes:
- keyless mode uses `open-meteo`.
- set `WEATHER_PROVIDER=openweather` only if you provide `WEATHER_API_KEY`.
- if live weather fetch fails, the app gracefully falls back to estimated local conditions so weather-assisted mode still works.

3. **Using Your Own Data**:
   Place your CSV file at `data/crop_yield_data.csv` with the schema described above, then run:
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
crop-yield-prediction/
│
├── data/
│   └── crop_yield_data.csv          # Input dataset (not included, generate or provide)
│
├── models/
│   └── crop_yield_rf_pipeline.joblib # Trained model (generated during execution)
│
├── outputs/
│   ├── correlation_heatmap.png
│   ├── feature_vs_yield_scatter.png
│   ├── yield_by_crop_type_boxplot.png
│   ├── yield_distribution.png
│   ├── feature_importance_TOP15.png
│   ├── feature_importance.csv
│   └── results_summary.json
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # CSV loading and validation
│   ├── eda.py                      # Exploratory data analysis & visualization
│   ├── preprocessing.py            # Feature scaling, encoding, imputation
│   ├── train.py                    # Model training & hyperparameter tuning
│   ├── evaluate.py                 # Evaluation metrics & reporting
│   ├── predict.py                  # Prediction & feature importance
│   └── utils.py                    # Utility functions
│
├── main.py                          # Main orchestration script
├── generate_sample_data.py          # Synthetic data generator
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 📈 Sample Input for Prediction

The model accepts predictions in the following format:

```python
sample_input = {
    "N": 90,                    # Nitrogen in kg/ha
    "P": 40,                    # Phosphorus in kg/ha
    "K": 40,                    # Potassium in kg/ha
    "pH": 6.5,                  # Soil pH
    "organic_matter": 2.8,      # Organic matter %
    "rainfall": 220,            # Annual rainfall mm
    "temp_min": 18,             # Min temperature °C
    "temp_max": 31,             # Max temperature °C
    "fertilizer_usage": 120,    # Fertilizer kg/ha
    "crop_type": "rice"         # Type of crop
}
```

**Predicted Output:** `~45.73 units` (example)

## 🎓 Evaluation Metrics

The model is evaluated using standard regression metrics:

| Metric | Formula | Interpretation |
|---|---|---|
| **R² Score** | 1 - (SS_res / SS_tot) | Proportion of variance explained (0-1, higher better) |
| **RMSE** | √(Σ(y_true - y_pred)² / n) | Average magnitude of prediction errors (lower better) |
| **MAE** | Σ\|y_true - y_pred\| / n | Mean absolute error (lower better) |
| **MAPE** | Σ\|\|y_true - y_pred\| / y_true\| / n × 100 | Percentage error (lower better) |

### Expected Performance (on synthetic data)
- **Train R²:** ~0.95+
- **Test R²:** ~0.92+
- **Test RMSE:** ~2.5 units
- **Test MAE:** ~1.8 units

*Note: Actual performance depends on data quality and completeness.*

## 🔬 Model Architecture

### Preprocessing Pipeline
```
Raw Data
   ↓
[Numerical Features] → Impute (median) → StandardScaler
   ↓
[Categorical Features] → Impute (frequent) → OneHotEncoder
   ↓
Combined Features
```

### Model: Random Forest Regressor
- **Algorithm:** Ensemble of Decision Trees
- **Hyperparameters Tuned:**
  - `n_estimators`: [100, 200]
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5]
- **Cross-Validation:** 5-fold
- **Optimization:** GridSearchCV with R² scoring

## 📊 Key Outputs

### 1. Visualizations (saved in `outputs/`)
- **correlation_heatmap.png** - Correlation matrix of all numerical features
- **feature_vs_yield_scatter.png** - Scatter plots with trend lines for each feature
- **yield_by_crop_type_boxplot.png** - Yield distribution by crop type
- **yield_distribution.png** - Histogram and KDE of target variable
- **feature_importance_TOP15.png** - Bar plot of top 15 important features

### 2. Data Files
- **feature_importance.csv** - Complete feature importance ranking
- **results_summary.json** - Model performance summary and configuration

### 3. Trained Model
- **models/crop_yield_rf_pipeline.joblib** - Ready-to-use model for predictions

## 🔧 Key Features

✅ **Data Handling**
- Robust CSV loading with validation
- Smart imputation (median for numerical, mode for categorical)
- Automatic handling of extra columns
- One-hot encoding for categorical variables

✅ **Exploratory Data Analysis**
- Data type and missing value analysis
- Summary statistics
- 5 high-quality visualizations
- Correlation and distribution analysis

✅ **Model Development**
- Modular preprocessing with ColumnTransformer
- Baseline and tuned models
- Comprehensive hyperparameter optimization
- 5-fold cross-validation

✅ **Evaluation**
- Multiple metrics (R², RMSE, MAE, MAPE)
- Train vs test comparison
- Overfitting detection
- Feature importance analysis

✅ **Prediction**
- Single-sample prediction function
- Batch prediction capability
- Input validation
- Interactive CLI for manual testing

✅ **Code Quality**
- Clean, modular architecture
- Comprehensive documentation
- Proper error handling
- Type hints
- Professional formatting

## 🎯 Usage Examples

### Example 1: Run Complete Pipeline
```bash
python main.py
```

### Example 2: Generate Data Only
```bash
python generate_sample_data.py
```

### Example 3: Use Trained Model for Predictions
```python
import joblib
from src.predict import predict_crop_yield

# Load trained model
model = joblib.load('models/crop_yield_rf_pipeline.joblib')

# Define feature names (from training)
feature_names = [...]  # See outputs/results_summary.json

# Make prediction
prediction = predict_crop_yield(model, {
    "N": 85, "P": 35, "K": 35, "pH": 6.8,
    "organic_matter": 2.5, "rainfall": 200,
    "temp_min": 16, "temp_max": 32,
    "fertilizer_usage": 110, "crop_type": "wheat"
}, feature_names)

print(f"Predicted Yield: {prediction:.2f}")
```

## 💡 Advanced Configuration

Edit `CONFIG` dictionary in `main.py` to customize:

```python
CONFIG = {
    'data_path': 'data/crop_yield_data.csv',    # Change data source
    'models_dir': 'models',                      # Model storage location
    'outputs_dir': 'outputs',                    # Output directory
    'test_size': 0.2,                            # Train-test split ratio
    'cv_folds': 5,                               # Cross-validation folds
    'random_state': 42,                          # Reproducibility seed
    # ... more settings
}
```

## 📚 Dependencies

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and preprocessing
- **matplotlib** - Visualization
- **seaborn** - Statistical visualization
- **joblib** - Model persistence

See `requirements.txt` for exact versions.

## ⚠️ Important Notes

1. **Data Format:** Ensure CSV contains all required columns with appropriate data types
2. **Memory Usage:** Large datasets (>100k samples) may require significant RAM
3. **Reproducibility:** Set `random_state` to the same value for consistent results
4. **Feature Names:** After one-hot encoding, categorical features become multiple binary columns
5. **Missing Values:** Handled automatically by the preprocessor

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| `FileNotFoundError: crop_yield_data.csv not found` | Run `python generate_sample_data.py` to create sample data |
| `Missing required columns` | Verify CSV has all columns listed in PROJECT schema |
| `Out of Memory` | Reduce dataset size or use a machine with more RAM |
| `Low R² Score` | Data quality may be poor; check for outliers and missing values |

## 📖 Model Interpretation

### Feature Importance
The `feature_importance.csv` file shows which variables most strongly influence yield. High-importance features should receive more attention in farm management.

### Prediction Output
The model outputs a continuous value representing predicted crop yield. To assess reliability:
- Check if prediction falls within historical data range
- Compare against baseline (mean yield)
- Use RMSE to understand expected error magnitude

## 🚀 Future Enhancements

Potential improvements:
- [ ] Try gradient boosting models (XGBoost, LightGBM)
- [ ] Add temporal patterns (year-over-year trends)
- [ ] Implement ensemble methods (stacking, voting)
- [ ] Add weather normalization
- [ ] Create web API for predictions
- [ ] Add SHAP for model interpretability
- [ ] Multi-regional model training

## 📄 License

This project is free to use for academic and commercial purposes.

## ✍️ Author Notes

This project demonstrates:
- Complete ML pipeline implementation
- Professional code organization
- Data science best practices
- Production-ready error handling
- Comprehensive documentation
- Academic rigor with practical applicability

Perfect for portfolio, coursework, or real-world agricultural analytics.

## 🤝 Support

For questions or issues:
1. Check the Troubleshooting section
2. Review data schema requirements
3. Examine generated logs and outputs
4. Verify all dependencies are installed

---

**Last Updated:** 2026-04-07  
**Version:** 1.0  
**Python Version:** 3.10+
