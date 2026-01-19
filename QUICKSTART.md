# Quick Start Guide

Get started with the ML/AI Learning Project in minutes!

## ⚡ 5-Minute Quick Start

### Step 1: Install Dependencies (1 minute)

```bash
# Clone the repository
git clone https://github.com/erikaramovich/dynamic_project_2026.git
cd dynamic_project_2026

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Generate Sample Data (30 seconds)

```bash
python src/ml_project/generate_data.py
```

Expected output:
```
✅ Data saved to: data/raw/housing_data.csv
```

### Step 3: Train Models (2 minutes)

```bash
python src/ml_project/train.py
```

Expected output:
```
✅ Linear Regression trained and saved
✅ Random Forest trained and saved
✅ Neural Network trained and saved

Best Model: Random Forest (RMSE: $40,292.65)
```

### Step 4: Make Predictions (30 seconds)

```bash
python src/ml_project/predict.py
```

Expected output:
```
House 1:
  Square Feet: 2,500
  Bedrooms: 4
  ...
Predicted Prices:
  Linear Regression    $  709,040.68
  Random Forest        $  655,196.09
  Neural Network       $  644,312.19
```

### Step 5: Visualize Results (1 minute)

```bash
python examples/visualize_example.py
```

This will show:
- Price distribution
- Feature correlations
- Model predictions vs actual
- Residual plots
- Feature importance

## 🎓 Next Steps

### Interactive Learning
```bash
jupyter notebook notebooks/ml_tutorial.ipynb
```

This opens an interactive notebook where you can:
- Run code step-by-step
- Modify parameters
- See immediate results
- Learn ML concepts hands-on

### Explore the Code

Start with these files in order:

1. **`src/ml_project/config.py`**
   - See all configuration options
   - Understand hyperparameters

2. **`src/ml_project/data_loader.py`**
   - Learn data loading
   - Understand validation

3. **`src/ml_project/preprocessing.py`**
   - Feature engineering
   - Data scaling

4. **`src/ml_project/models.py`**
   - Three different models
   - Model implementations

5. **`src/ml_project/evaluate.py`**
   - Model evaluation metrics
   - Performance comparison

## 🧪 Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/ml_project --cov-report=html
```

## 🎯 Common Tasks

### Change Model Hyperparameters

Edit `src/ml_project/config.py`:

```python
# Random Forest settings
RF_N_ESTIMATORS = 200  # Changed from 100
RF_MAX_DEPTH = 30      # Changed from 20

# Neural Network settings
NN_EPOCHS = 150        # Changed from 100
NN_HIDDEN_LAYERS = [128, 64, 32]  # Changed architecture
```

Then retrain:
```bash
python src/ml_project/train.py
```

### Generate Different Data

```python
# In src/ml_project/generate_data.py, modify:
data = generate_housing_data(n_samples=2000, seed=123)  # More samples, different seed
```

### Add New Features

Edit the `generate_housing_data()` function in `src/ml_project/generate_data.py` to add features like:
- Swimming pool
- School rating
- Distance to downtown
- Crime rate

### Try Different Test Sizes

In `src/ml_project/config.py`:
```python
TEST_SIZE = 0.3  # Changed from 0.2 (30% for testing)
```

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### TensorFlow Warnings
These are normal:
```
Could not find cuda drivers on your machine, GPU will not be used.
```

To suppress:
```bash
export TF_CPP_MIN_LOG_LEVEL=2
```

### Model Not Found
```bash
# Make sure you've trained models first
python src/ml_project/train.py
```

### Data Not Found
```bash
# Generate data first
python src/ml_project/generate_data.py
```

## 📊 Understanding the Output

### Training Output

```
Linear Regression: RMSE = 49,004, R² = 0.6853
Random Forest:     RMSE = 40,293, R² = 0.7873
Neural Network:    RMSE = 73,186, R² = 0.2981
```

**What this means:**
- **RMSE** (Root Mean Squared Error): Average prediction error in dollars. Lower is better.
- **R²** (R-squared): How well the model explains variance. Closer to 1.0 is better.
- **Random Forest wins**: Lowest RMSE, highest R²

### Prediction Summary

```
Mean % Error:            5.63%
Predictions within 10%:  84.0%
Predictions within 20%:  97.0%
```

**What this means:**
- Average prediction is off by 5.63%
- 84% of predictions are within 10% of actual price
- Model is quite accurate!

## 🎨 Customization Ideas

### Beginner Level
1. Change random seed and compare results
2. Modify feature importance plot to show more/fewer features
3. Try different test/train split ratios
4. Change visualization colors

### Intermediate Level
1. Add cross-validation
2. Implement grid search for hyperparameters
3. Add new evaluation metrics (MAPE, MAE)
4. Create ensemble of models

### Advanced Level
1. Add XGBoost or LightGBM models
2. Implement SHAP for model interpretability
3. Create REST API for predictions
4. Add automatic hyperparameter tuning
5. Deploy model to cloud

## 📚 Learning Resources

See [LEARNING_RESOURCES.md](LEARNING_RESOURCES.md) for:
- Books and courses
- Online tutorials
- Practice platforms
- Community resources

## 🤝 Getting Help

1. **Check Documentation**: README.md has detailed information
2. **Read Error Messages**: They often tell you what's wrong
3. **Search Issues**: Someone might have had the same problem
4. **Ask Questions**: Create a GitHub issue

## 🎉 You're Ready!

You now have a working ML project. Start experimenting, break things, and learn!

Key learning points:
- ✅ Complete ML pipeline
- ✅ Multiple model types
- ✅ Model evaluation
- ✅ Data preprocessing
- ✅ Visualization

**Next challenge**: Modify the code to solve a different prediction problem!

---

**Happy Learning! 🚀**
