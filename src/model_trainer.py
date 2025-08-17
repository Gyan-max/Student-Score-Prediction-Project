"""
Model training module for student score prediction.
Handles model training, evaluation, and persistence.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import os


class ModelTrainer:
    """Handles machine learning model training and evaluation."""
    
    def __init__(self):
        self.model = None
        self.feature_names = ['Hours_Studied', 'Attendance']
        self.is_trained = False
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
        """
        Train a linear regression model on the training data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target values
            
        Returns:
            LinearRegression: Trained model
        """
        print("Training Linear Regression model...")
        
        # Validate input data
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data is empty")
        
        if len(X_train) != len(y_train):
            raise ValueError("Feature and target arrays have different lengths")
        
        # Create and train the model
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print(f"Model training completed with {len(X_train)} samples")
        
        # Display model coefficients
        self._display_model_info()
        
        return self.model
    
    def get_model_coefficients(self) -> dict:
        """
        Extract and interpret model coefficients.
        
        Returns:
            dict: Model coefficients and intercept
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained first")
        
        coefficients = {
            'intercept': self.model.intercept_,
            'coefficients': {}
        }
        
        for i, feature in enumerate(self.feature_names):
            coefficients['coefficients'][feature] = self.model.coef_[i]
        
        return coefficients
    
    def _display_model_info(self):
        """Display model coefficients and interpretation."""
        if not self.is_trained:
            return
        
        coeffs = self.get_model_coefficients()
        
        print("\nModel Coefficients:")
        print("=" * 30)
        print(f"Intercept: {coeffs['intercept']:.3f}")
        
        for feature, coeff in coeffs['coefficients'].items():
            print(f"{feature}: {coeff:.3f}")
        
        print("\nModel Interpretation:")
        print("-" * 20)
        print(f"Base score (when both features are 0): {coeffs['intercept']:.1f}")
        
        hours_coeff = coeffs['coefficients']['Hours_Studied']
        attendance_coeff = coeffs['coefficients']['Attendance']
        
        print(f"Each additional hour of study increases score by: {hours_coeff:.2f} points")
        print(f"Each 1% increase in attendance increases score by: {attendance_coeff:.2f} points")
        
        # Model equation
        print(f"\nModel Equation:")
        print(f"Final_Score = {coeffs['intercept']:.2f} + "
              f"{hours_coeff:.2f} × Hours_Studied + "
              f"{attendance_coeff:.2f} × Attendance")
    
    def save_model(self, file_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            file_path (str): Path to save the model
        """
        if not self.is_trained or self.model is None:
            raise ValueError("No trained model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'coefficients': self.get_model_coefficients()
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {file_path}")
    
    def load_model(self, file_path: str) -> LinearRegression:
        """
        Load a trained model from disk.
        
        Args:
            file_path (str): Path to the saved model
            
        Returns:
            LinearRegression: Loaded model
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print(f"Model loaded from: {file_path}")
        return self.model    

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target values
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained first")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        metrics = {
            'r2_score': r2,
            'mean_absolute_error': mae,
            'root_mean_squared_error': rmse,
            'predictions': y_pred,
            'actual': y_test.values
        }
        
        # Display evaluation results
        self._display_evaluation_results(metrics)
        
        return metrics
    
    def _display_evaluation_results(self, metrics: dict) -> None:
        """
        Display model evaluation results with interpretation.
        
        Args:
            metrics (dict): Evaluation metrics
        """
        print("\nModel Evaluation Results:")
        print("=" * 40)
        
        r2 = metrics['r2_score']
        mae = metrics['mean_absolute_error']
        rmse = metrics['root_mean_squared_error']
        
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Error: {mae:.2f} points")
        print(f"Root Mean Squared Error: {rmse:.2f} points")
        
        print("\nMetrics Interpretation:")
        print("-" * 25)
        
        # R² interpretation
        if r2 >= 0.9:
            r2_quality = "Excellent"
        elif r2 >= 0.8:
            r2_quality = "Very Good"
        elif r2 >= 0.7:
            r2_quality = "Good"
        elif r2 >= 0.5:
            r2_quality = "Moderate"
        else:
            r2_quality = "Poor"
        
        print(f"R² Score ({r2_quality}): The model explains {r2*100:.1f}% of the variance in final scores")
        
        # MAE interpretation
        print(f"Mean Absolute Error: On average, predictions are off by {mae:.1f} points")
        
        # RMSE interpretation
        print(f"Root Mean Squared Error: {rmse:.1f} points (penalizes larger errors more)")
        
        # Performance assessment
        if mae <= 5:
            performance = "Excellent"
        elif mae <= 10:
            performance = "Good"
        elif mae <= 15:
            performance = "Fair"
        else:
            performance = "Needs Improvement"
        
        print(f"\nOverall Model Performance: {performance}")
        
        # Additional insights
        predictions = metrics['predictions']
        actual = metrics['actual']
        
        # Calculate prediction accuracy within ranges
        within_5 = np.sum(np.abs(predictions - actual) <= 5) / len(actual) * 100
        within_10 = np.sum(np.abs(predictions - actual) <= 10) / len(actual) * 100
        
        print(f"\nPrediction Accuracy:")
        print(f"Within 5 points: {within_5:.1f}% of predictions")
        print(f"Within 10 points: {within_10:.1f}% of predictions")
        
        # Show some example predictions
        print(f"\nSample Predictions (first 5):")
        for i in range(min(5, len(predictions))):
            print(f"Actual: {actual[i]:.1f}, Predicted: {predictions[i]:.1f}, "
                  f"Error: {abs(actual[i] - predictions[i]):.1f}")
    
    def predict_single(self, hours_studied: float, attendance: float) -> float:
        """
        Make a single prediction.
        
        Args:
            hours_studied (float): Hours studied
            attendance (float): Attendance percentage
            
        Returns:
            float: Predicted final score
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained first")
        
        # Create input DataFrame with proper feature names
        X = pd.DataFrame({
            'Hours_Studied': [hours_studied],
            'Attendance': [attendance]
        })
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        return prediction