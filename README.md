# Student Score Prediction Project

A machine learning project that predicts student final exam scores based on study habits (hours studied and ate percentage) using Linear Regression.

## 🎯 Project Overview

- **Objective**: Predict student final exam scores using study hours and attendance data
- **Method**: Linear Regression
- **Features**: Hours_Studied, Attendance
- **Target**: Final_Score
- **Interface**: Interactive web dashboard built with Streamlit

## 📁 Project Structure

```
new project/
├── data/
│   └── student_data.csv          # Dataset with 100 student records
├── src/
│   ├── __init__.py
│   ├── data_processor.py         # Data loading and preprocessing
│   ├── visualizer.py             # Data visualization functions
│   ├── model_trainer.py          # Model training and evaluation
│   └── predictor.py              # Prediction functionality
├── models/
│   └── student_score_model.pkl   # Trained model (generated)
├── notebooks/
│   └── student_score_analysis.ipynb  # Jupyter notebook analysis
├── dashboard.py                  # Interactive web dashboard
├── launch_dashboard.py           # Easy launcher script
├── run_pipeline.py               # Training pipeline runner
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python run_pipeline.py
```

### 4. Launch the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in your web browser at `http://localhost:8501`

## 📊 Dashboard Features

### Interactive Prediction Tool
- **Sliders**: Adjust study hours (0-12) and attendance (0-100%)
- **Real-time Predictions**: Get instant score predictions
- **Performance Categories**: A, B, C, D, F grade classifications
- **Score Breakdown**: See how each factor contributes to the final score

### Data Visualization
- **Dataset Statistics**: Summary statistics of the training data
- **Correlation Heatmap**: Visual correlation matrix
- **Scatter Plots**: Relationship between features and target
- **Scenario Comparison**: Compare different study habit scenarios

### Recommendations
- Personalized suggestions based on input values
- Performance improvement tips
- Study habit optimization advice

## 🔧 Alternative Usage

### Command Line Interface
```bash
python run_pipeline.py  # Train the model
python launch_dashboard.py  # Launch dashboard with auto-training
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/student_score_analysis.ipynb
```

### Python API
```python
from src.predictor import ScorePredictor

predictor = ScorePredictor()
predictor.load_model('models/student_score_model.pkl')
score = predictor.predict_score(hours_studied=4, attendance=80)
print(f"Predicted score: {score:.1f}")
```

## 📈 Model Performance

The trained Linear Regression model achieves:
- **R² Score**: ~0.7-0.8 (explains 70-80% of variance)
- **Mean Absolute Error**: ~8-12 points
- **Features**: Strong positive correlation between both study hours and attendance with final scores

## 🎓 Example Predictions

| Hours Studied | Attendance | Predicted Score | Grade |
|---------------|------------|-----------------|-------|
| 4.0           | 80%        | ~75.0           | C     |
| 6.0           | 90%        | ~85.0           | B     |
| 8.0           | 95%        | ~92.0           | A     |
| 2.0           | 60%        | ~55.0           | F     |

## 🛠️ Technical Details

### Data Processing
- Handles missing values and outliers
- Validates input ranges
- Splits data into 80% training, 20% testing

### Model Training
- Uses scikit-learn LinearRegression
- Evaluates with R², MAE, and RMSE metrics
- Saves model using pickle for persistence

### Prediction System
- Input validation for realistic ranges
- Confidence intervals and explanations
- Batch prediction capabilities

## 📋 Requirements

- Python 3.7+
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly
- streamlit (for dashboard)
- jupyter (for notebook analysis)

## 🧹 Project Maintenance

This project maintains a clean structure by:
- Excluding virtual environments (use `.gitignore`)
- Separating source code into modules (`src/` directory)
- Keeping data and models in dedicated directories
- Providing multiple interfaces for different use cases

### Setting Up Development Environment

```bash
# Clone/download the project
# Navigate to project directory
cd "new project"

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the project
python launch_dashboard.py
```

## 🤝 Contributing

Feel free to contribute by:
- Adding new features to the dashboard
- Improving model performance
- Adding more visualization options
- Enhancing the user interface

## 📄 License

This project is for educational purposes and demonstrates machine learning concepts in student performance prediction.# Student-Score-Prediction-Project
