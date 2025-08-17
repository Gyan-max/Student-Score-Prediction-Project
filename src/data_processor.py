"""
Data processing module for student score prediction.
Handles data loading, validation, and preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


class DataProcessor:
    """Handles all data processing operations for the student score prediction model."""
    
    def __init__(self):
        self.required_columns = ['Hours_Studied', 'Attendance', 'Final_Score']
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load student data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            df = pd.read_csv(file_path)
            
            if df.empty:
                raise ValueError("CSV file is empty")
            
            # Validate required columns exist
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            print(f"Successfully loaded {len(df)} records from {file_path}")
            return df
            
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty or corrupted")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error loading data: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality and ranges.
        
        Args:
            df (pd.DataFrame): Data to validate
            
        Returns:
            bool: True if data is valid
            
        Raises:
            ValueError: If data validation fails
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check for required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check data types and convert if necessary
        for col in self.required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    raise ValueError(f"Cannot convert column {col} to numeric")
        
        # Check for valid ranges
        if (df['Hours_Studied'] < 0).any() or (df['Hours_Studied'] > 24).any():
            print("Warning: Hours_Studied values outside expected range (0-24)")
        
        if (df['Attendance'] < 0).any() or (df['Attendance'] > 100).any():
            print("Warning: Attendance values outside expected range (0-100)")
        
        if (df['Final_Score'] < 0).any() or (df['Final_Score'] > 100).any():
            print("Warning: Final_Score values outside expected range (0-100)")
        
        # Check for missing values
        missing_count = df[self.required_columns].isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: Found {missing_count} missing values")
        
        print("Data validation completed successfully")
        return True
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            df (pd.DataFrame): Raw data to clean
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        df_clean = df.copy()
        
        # Convert to numeric types
        for col in self.required_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Handle missing values
        missing_before = df_clean[self.required_columns].isnull().sum().sum()
        if missing_before > 0:
            print(f"Handling {missing_before} missing values...")
            
            # If less than 5% missing, drop rows; otherwise impute with median
            missing_percentage = missing_before / (len(df_clean) * len(self.required_columns))
            
            if missing_percentage < 0.05:
                df_clean = df_clean.dropna(subset=self.required_columns)
                print(f"Dropped rows with missing values")
            else:
                for col in self.required_columns:
                    if df_clean[col].isnull().any():
                        median_val = df_clean[col].median()
                        df_clean[col].fillna(median_val, inplace=True)
                        print(f"Imputed missing values in {col} with median: {median_val}")
        
        # Remove extreme outliers using IQR method
        df_clean = self._remove_outliers(df_clean)
        
        # Ensure data is within valid ranges
        df_clean['Hours_Studied'] = df_clean['Hours_Studied'].clip(0, 24)
        df_clean['Attendance'] = df_clean['Attendance'].clip(0, 100)
        df_clean['Final_Score'] = df_clean['Final_Score'].clip(0, 100)
        
        print(f"Data cleaning completed. Records: {len(df)} -> {len(df_clean)}")
        return df_clean
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove extreme outliers using IQR method.
        
        Args:
            df (pd.DataFrame): Data to process
            
        Returns:
            pd.DataFrame: Data with outliers removed
        """
        df_no_outliers = df.copy()
        outliers_removed = 0
        
        for col in self.required_columns:
            Q1 = df_no_outliers[col].quantile(0.25)
            Q3 = df_no_outliers[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds (more conservative - 2.5 * IQR)
            lower_bound = Q1 - 2.5 * IQR
            upper_bound = Q3 + 2.5 * IQR
            
            # Count outliers before removal
            outliers_count = ((df_no_outliers[col] < lower_bound) | 
                            (df_no_outliers[col] > upper_bound)).sum()
            
            if outliers_count > 0:
                df_no_outliers = df_no_outliers[
                    (df_no_outliers[col] >= lower_bound) & 
                    (df_no_outliers[col] <= upper_bound)
                ]
                outliers_removed += outliers_count
                print(f"Removed {outliers_count} outliers from {col}")
        
        if outliers_removed > 0:
            print(f"Total outliers removed: {outliers_removed}")
        
        return df_no_outliers
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        Split data into training and testing sets.
        
        Args:
            df (pd.DataFrame): Clean data to split
            test_size (float): Proportion of data for testing (default: 0.2)
            random_state (int): Random state for reproducibility (default: 42)
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Separate features and target
        features = ['Hours_Studied', 'Attendance']
        target = 'Final_Score'
        
        X = df[features]
        y = df[target]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Validate splits
        self._validate_splits(X_train, X_test, y_train, y_test)
        
        print(f"Data split completed:")
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Testing set: {len(X_test)} samples")
        print(f"  Test size: {test_size:.1%}")
        
        return X_train, X_test, y_train, y_test
    
    def _validate_splits(self, X_train, X_test, y_train, y_test):
        """
        Validate that train/test splits contain representative samples.
        
        Args:
            X_train, X_test, y_train, y_test: Split datasets
        """
        # Check that both sets have data
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("One of the splits is empty")
        
        # Check feature distributions are similar
        for feature in X_train.columns:
            train_mean = X_train[feature].mean()
            test_mean = X_test[feature].mean()
            
            # Allow up to 20% difference in means
            if abs(train_mean - test_mean) / train_mean > 0.2:
                print(f"Warning: Large difference in {feature} means between train/test sets")
        
        # Check target distributions
        train_target_mean = y_train.mean()
        test_target_mean = y_test.mean()
        
        if abs(train_target_mean - test_target_mean) / train_target_mean > 0.2:
            print("Warning: Large difference in target means between train/test sets")
        
        print("Split validation completed")