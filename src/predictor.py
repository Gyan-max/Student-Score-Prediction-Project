"""
Prediction module for student score prediction.
Handles loading trained models and making predictions.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression


class ScorePredictor:
    """Handles score predictions using trained models."""
    
    def __init__(self):
        self.model = None
        self.feature_names = ['Hours_Studied', 'Attendance']
        self.model_loaded = False
    
    def load_model(self, model_path: str) -> LinearRegression:
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model file
            
        Returns:
            LinearRegression: Loaded model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is corrupted
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.model_loaded = True
            
            print(f"Model successfully loaded from: {model_path}")
            
            # Display model info
            coeffs = model_data['coefficients']
            print(f"Model intercept: {coeffs['intercept']:.3f}")
            for feature, coeff in coeffs['coefficients'].items():
                print(f"{feature} coefficient: {coeff:.3f}")
            
            return self.model
            
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
    
    def predict_score(self, hours_studied: float, attendance: float) -> float:
        """
        Predict final score for a single student.
        
        Args:
            hours_studied (float): Number of hours studied
            attendance (float): Attendance percentage
            
        Returns:
            float: Predicted final score
            
        Raises:
            ValueError: If model not loaded or invalid inputs
        """
        if not self.model_loaded or self.model is None:
            raise ValueError("Model must be loaded first. Use load_model() method.")
        
        # Validate inputs
        self.validate_input(hours_studied, attendance)
        
        # Create input DataFrame with proper feature names
        X = pd.DataFrame({
            'Hours_Studied': [hours_studied],
            'Attendance': [attendance]
        })
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Ensure prediction is within valid range
        prediction = max(0, min(100, prediction))
        
        return prediction
    
    def predict_batch(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for multiple students.
        
        Args:
            data (pd.DataFrame): DataFrame with Hours_Studied and Attendance columns
            
        Returns:
            np.ndarray: Array of predicted scores
            
        Raises:
            ValueError: If model not loaded or invalid data format
        """
        if not self.model_loaded or self.model is None:
            raise ValueError("Model must be loaded first. Use load_model() method.")
        
        # Validate data format
        required_columns = ['Hours_Studied', 'Attendance']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate each row
        for idx, row in data.iterrows():
            try:
                self.validate_input(row['Hours_Studied'], row['Attendance'])
            except ValueError as e:
                print(f"Warning: Row {idx} has invalid data: {e}")
        
        # Make predictions
        X = data[required_columns].values
        predictions = self.model.predict(X)
        
        # Ensure predictions are within valid range
        predictions = np.clip(predictions, 0, 100)
        
        return predictions
    
    def predict_with_confidence(self, hours_studied: float, attendance: float) -> dict:
        """
        Make prediction with additional confidence information.
        
        Args:
            hours_studied (float): Number of hours studied
            attendance (float): Attendance percentage
            
        Returns:
            dict: Prediction with confidence metrics
        """
        prediction = self.predict_score(hours_studied, attendance)
        
        # Calculate feature contributions
        if self.model_loaded:
            intercept = self.model.intercept_
            hours_coeff = self.model.coef_[0]
            attendance_coeff = self.model.coef_[1]
            
            hours_contribution = hours_coeff * hours_studied
            attendance_contribution = attendance_coeff * attendance
            
            result = {
                'predicted_score': prediction,
                'base_score': intercept,
                'hours_contribution': hours_contribution,
                'attendance_contribution': attendance_contribution,
                'breakdown': {
                    'base': f"{intercept:.1f} points (baseline)",
                    'hours': f"{hours_contribution:+.1f} points from {hours_studied} hours studied",
                    'attendance': f"{attendance_contribution:+.1f} points from {attendance}% attendance"
                }
            }
            
            return result
        
        return {'predicted_score': prediction}
    
    def validate_input(self, hours_studied: float, attendance: float) -> bool:
        """
        Validate input parameters for prediction.
        
        Args:
            hours_studied (float): Number of hours studied
            attendance (float): Attendance percentage
            
        Returns:
            bool: True if inputs are valid
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Check if inputs are numeric
        try:
            hours_studied = float(hours_studied)
            attendance = float(attendance)
        except (ValueError, TypeError):
            raise ValueError("Hours studied and attendance must be numeric values")
        
        # Check if inputs are not NaN or infinite
        if np.isnan(hours_studied) or np.isinf(hours_studied):
            raise ValueError("Hours studied cannot be NaN or infinite")
        
        if np.isnan(attendance) or np.isinf(attendance):
            raise ValueError("Attendance cannot be NaN or infinite")
        
        # Check valid ranges
        if hours_studied < 0:
            raise ValueError("Hours studied cannot be negative")
        
        if hours_studied > 24:
            raise ValueError("Hours studied cannot exceed 24 hours per day")
        
        if attendance < 0:
            raise ValueError("Attendance percentage cannot be negative")
        
        if attendance > 100:
            raise ValueError("Attendance percentage cannot exceed 100%")
        
        # Warnings for unusual but valid values
        if hours_studied > 12:
            print(f"Warning: {hours_studied} hours of study per day is unusually high")
        
        if attendance < 30:
            print(f"Warning: {attendance}% attendance is very low")
        
        return True
    
    def get_prediction_explanation(self, hours_studied: float, attendance: float) -> str:
        """
        Get a human-readable explanation of the prediction.
        
        Args:
            hours_studied (float): Number of hours studied
            attendance (float): Attendance percentage
            
        Returns:
            str: Explanation of the prediction
        """
        if not self.model_loaded:
            return "Model not loaded. Cannot provide explanation."
        
        try:
            result = self.predict_with_confidence(hours_studied, attendance)
            prediction = result['predicted_score']
            
            explanation = f"Prediction for student with {hours_studied} hours studied and {attendance}% attendance:\n\n"
            explanation += f"Predicted Final Score: {prediction:.1f}/100\n\n"
            explanation += "Score Breakdown:\n"
            
            for component, description in result['breakdown'].items():
                explanation += f"  • {description}\n"
            
            explanation += f"\nTotal: {prediction:.1f} points\n\n"
            
            # Performance category
            if prediction >= 90:
                category = "Excellent (A)"
            elif prediction >= 80:
                category = "Good (B)"
            elif prediction >= 70:
                category = "Satisfactory (C)"
            elif prediction >= 60:
                category = "Needs Improvement (D)"
            else:
                category = "Unsatisfactory (F)"
            
            explanation += f"Performance Category: {category}\n"
            
            # Recommendations
            explanation += "\nRecommendations:\n"
            if hours_studied < 5:
                explanation += "  • Consider increasing study hours for better performance\n"
            if attendance < 80:
                explanation += "  • Improve attendance to boost final score\n"
            if prediction < 70:
                explanation += "  • Both study hours and attendance need improvement\n"
            elif prediction >= 85:
                explanation += "  • Excellent study habits! Keep up the good work\n"
            
            return explanation
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def compare_scenarios(self, base_hours: float, base_attendance: float, 
                         scenarios: list) -> pd.DataFrame:
        """
        Compare prediction results for different scenarios.
        
        Args:
            base_hours (float): Base hours studied
            base_attendance (float): Base attendance
            scenarios (list): List of (hours, attendance) tuples to compare
            
        Returns:
            pd.DataFrame: Comparison results
        """
        if not self.model_loaded:
            raise ValueError("Model must be loaded first")
        
        results = []
        
        # Add base scenario
        base_pred = self.predict_score(base_hours, base_attendance)
        results.append({
            'Scenario': 'Base',
            'Hours_Studied': base_hours,
            'Attendance': base_attendance,
            'Predicted_Score': base_pred,
            'Score_Change': 0.0
        })
        
        # Add comparison scenarios
        for i, (hours, attendance) in enumerate(scenarios, 1):
            pred = self.predict_score(hours, attendance)
            change = pred - base_pred
            
            results.append({
                'Scenario': f'Scenario {i}',
                'Hours_Studied': hours,
                'Attendance': attendance,
                'Predicted_Score': pred,
                'Score_Change': change
            })
        
        df = pd.DataFrame(results)
        df['Predicted_Score'] = df['Predicted_Score'].round(1)
        df['Score_Change'] = df['Score_Change'].round(1)
        
        return df