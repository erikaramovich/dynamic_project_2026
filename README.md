# ML/AI Learning Project 2026 🚀

A comprehensive Machine Learning and Artificial Intelligence project designed for junior engineers to learn modern ML concepts, tools, and best practices in 2026.

## 🎯 Learning Objectives

This project will help you understand:
- **Data preprocessing and feature engineering**
- **Multiple ML algorithms** (Linear Regression, Random Forest, Neural Networks)
- **Model evaluation and comparison**
- **Modern Python ML tools** (scikit-learn, TensorFlow/Keras, pandas, numpy)
- **ML project structure and best practices**
- **Model persistence and deployment concepts**
- **Visualization and interpretation** of ML results

## 📋 Project Overview

This project implements a **House Price Prediction System** using multiple machine learning approaches. It demonstrates the complete ML workflow from data loading to model deployment.

### Key Features:
- 🔄 **Complete ML Pipeline**: Data loading → Preprocessing → Training → Evaluation → Prediction
- 📊 **Multiple Models**: Compare Linear Regression, Random Forest, and Neural Networks
- 📈 **Visualization**: Interactive plots and model performance charts
- 🧪 **Testing**: Comprehensive unit and integration tests
- 📓 **Interactive Learning**: Jupyter notebooks for hands-on experimentation
- 🛠️ **Modern Tools**: Uses 2026 best practices and latest ML libraries

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/erikaramovich/dynamic_project_2026.git
cd dynamic_project_2026
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📚 Usage

### 1. Generate Sample Data
```bash
python src/ml_project/generate_data.py
```

### 2. Train Models
```bash
python src/ml_project/train.py
```

### 3. Make Predictions
```bash
python src/ml_project/predict.py
```

### 4. Interactive Learning with Jupyter
```bash
jupyter notebook notebooks/ml_tutorial.ipynb
```

## 📁 Project Structure

```
dynamic_project_2026/
├── src/ml_project/          # Main source code
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   ├── data_loader.py       # Data loading utilities
│   ├── preprocessing.py     # Data preprocessing
│   ├── models.py            # Model implementations
│   ├── train.py             # Training script
│   ├── predict.py           # Prediction script
│   ├── evaluate.py          # Model evaluation
│   ├── visualize.py         # Visualization utilities
│   └── generate_data.py     # Sample data generation
├── data/                    # Data directory
│   ├── raw/                 # Raw data
│   └── processed/           # Processed data
├── models/                  # Saved models
├── notebooks/               # Jupyter notebooks
│   └── ml_tutorial.ipynb   # Interactive tutorial
├── tests/                   # Unit tests
├── logs/                    # Application logs
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🧠 ML Concepts Covered

### 1. **Data Preprocessing**
- Feature scaling and normalization
- Handling missing values
- Train-test splitting
- Feature engineering

### 2. **Algorithms**
- **Linear Regression**: Simple, interpretable baseline model
- **Random Forest**: Ensemble learning with decision trees
- **Neural Networks**: Deep learning with TensorFlow/Keras

### 3. **Model Evaluation**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Cross-validation

### 4. **Visualization**
- Feature correlation heatmaps
- Prediction vs actual plots
- Model performance comparison
- Feature importance analysis

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run tests with coverage:
```bash
pytest tests/ --cov=src/ml_project --cov-report=html
```

## 📊 Example Results

After training, you'll see model comparison:
```
Model Performance Comparison:
----------------------------
Linear Regression: RMSE = 45,230, R² = 0.85
Random Forest:     RMSE = 32,150, R² = 0.92
Neural Network:    RMSE = 30,890, R² = 0.93
```

## 🛠️ Technologies Used

- **Python 3.9+**: Programming language
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: Traditional ML algorithms
- **TensorFlow/Keras**: Deep learning
- **Matplotlib/Seaborn**: Data visualization
- **Plotly**: Interactive visualizations
- **Jupyter**: Interactive notebooks
- **Pytest**: Testing framework

## 📖 Learning Path

### For Beginners:
1. Start with `notebooks/ml_tutorial.ipynb` for interactive learning
2. Read through `src/ml_project/data_loader.py` to understand data handling
3. Explore `src/ml_project/preprocessing.py` for feature engineering
4. Study `src/ml_project/models.py` to see different ML implementations

### For Intermediate Learners:
1. Experiment with different hyperparameters in `config.py`
2. Add new features in preprocessing pipeline
3. Try different model architectures in the neural network
4. Implement cross-validation

### For Advanced Learners:
1. Add new model types (XGBoost, LightGBM)
2. Implement hyperparameter tuning with GridSearch
3. Add model explainability (SHAP values)
4. Create a REST API for model serving

## 🔧 Configuration

Edit `src/ml_project/config.py` to customize:
- Model hyperparameters
- Data paths
- Training settings
- Random seeds for reproducibility

## 📝 Best Practices Demonstrated

- ✅ **Code organization**: Modular, reusable components
- ✅ **Documentation**: Clear comments and docstrings
- ✅ **Testing**: Unit tests for critical functions
- ✅ **Version control**: Git with proper .gitignore
- ✅ **Reproducibility**: Random seeds and configuration files
- ✅ **Logging**: Proper logging for debugging
- ✅ **Error handling**: Robust error handling

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root and virtual environment is activated
pip install -r requirements.txt
```

### Model Training Fails
- Check that sample data exists: `python src/ml_project/generate_data.py`
- Verify data format in `data/raw/housing_data.csv`

### TensorFlow Warnings
- TensorFlow may show info messages; these are normal and can be ignored
- To reduce output, set: `export TF_CPP_MIN_LOG_LEVEL=2`

## 🌟 Next Steps

1. **Experiment**: Try different parameters and models
2. **Real Data**: Use actual datasets from Kaggle or UCI ML Repository
3. **Deploy**: Create a web interface with Flask or FastAPI
4. **Scale**: Learn about distributed training and cloud deployment
5. **Specialize**: Dive deeper into Computer Vision, NLP, or Reinforcement Learning

## 📚 Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Fast.ai Courses](https://www.fast.ai/)

## 🤝 Contributing

This is a learning project. Feel free to:
- Add new features
- Improve documentation
- Fix bugs
- Add more examples

## 📄 License

This project is for educational purposes.

## 👨‍💻 About

Created as a learning project for junior engineers starting their journey in Machine Learning and Artificial Intelligence in 2026.

---

**Happy Learning! 🎓✨**