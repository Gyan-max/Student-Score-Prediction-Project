"""
Visualization module for student score prediction.
Creates exploratory data analysis plots and charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class DataVisualizer:
    """Handles all data visualization operations for the student score prediction model."""
    
    def __init__(self):
        # Set style for better-looking plots
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_scatter_relationships(self, df: pd.DataFrame) -> None:
        """
        Create scatter plots showing relationships between features and target.
        
        Args:
            df (pd.DataFrame): Data to visualize
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Hours_Studied vs Final_Score
        axes[0].scatter(df['Hours_Studied'], df['Final_Score'], alpha=0.6, color='blue')
        axes[0].set_xlabel('Hours Studied')
        axes[0].set_ylabel('Final Score')
        axes[0].set_title('Hours Studied vs Final Score')
        axes[0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['Hours_Studied'], df['Final_Score'], 1)
        p = np.poly1d(z)
        axes[0].plot(df['Hours_Studied'], p(df['Hours_Studied']), "r--", alpha=0.8)
        
        # Attendance vs Final_Score
        axes[1].scatter(df['Attendance'], df['Final_Score'], alpha=0.6, color='green')
        axes[1].set_xlabel('Attendance (%)')
        axes[1].set_ylabel('Final Score')
        axes[1].set_title('Attendance vs Final Score')
        axes[1].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['Attendance'], df['Final_Score'], 1)
        p = np.poly1d(z)
        axes[1].plot(df['Attendance'], p(df['Attendance']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate and display correlations
        corr_hours = df['Hours_Studied'].corr(df['Final_Score'])
        corr_attendance = df['Attendance'].corr(df['Final_Score'])
        
        print(f"Correlation between Hours Studied and Final Score: {corr_hours:.3f}")
        print(f"Correlation between Attendance and Final Score: {corr_attendance:.3f}")
    
    def plot_distribution(self, df: pd.DataFrame) -> None:
        """
        Plot distribution of all variables.
        
        Args:
            df (pd.DataFrame): Data to visualize
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Hours_Studied distribution
        axes[0, 0].hist(df['Hours_Studied'], bins=15, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('Hours Studied')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Hours Studied')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Attendance distribution
        axes[0, 1].hist(df['Attendance'], bins=15, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_xlabel('Attendance (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Attendance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Final_Score distribution
        axes[1, 0].hist(df['Final_Score'], bins=15, alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_xlabel('Final Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Final Scores')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot for all variables
        df_normalized = df[['Hours_Studied', 'Attendance', 'Final_Score']].copy()
        # Normalize Hours_Studied to 0-100 scale for comparison
        df_normalized['Hours_Studied'] = (df_normalized['Hours_Studied'] / 24) * 100
        
        axes[1, 1].boxplot([df_normalized['Hours_Studied'], 
                           df_normalized['Attendance'], 
                           df_normalized['Final_Score']], 
                          labels=['Hours Studied\n(scaled)', 'Attendance', 'Final Score'])
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Box Plot Comparison (Hours Scaled to 0-100)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()   
 
    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """
        Create correlation heatmap of all variables.
        
        Args:
            df (pd.DataFrame): Data to visualize
        """
        # Calculate correlation matrix
        corr_matrix = df[['Hours_Studied', 'Attendance', 'Final_Score']].corr()
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Correlation Matrix: Study Habits vs Final Score')
        plt.tight_layout()
        plt.show()
        
        # Print correlation insights
        print("\nCorrelation Analysis:")
        print("=" * 40)
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) > 0.7:
                    strength = "Strong"
                elif abs(corr_val) > 0.5:
                    strength = "Moderate"
                elif abs(corr_val) > 0.3:
                    strength = "Weak"
                else:
                    strength = "Very weak"
                
                direction = "positive" if corr_val > 0 else "negative"
                print(f"{col1} vs {col2}: {corr_val:.3f} ({strength} {direction})")
    
    def display_summary_stats(self, df: pd.DataFrame) -> None:
        """
        Display descriptive statistics for the dataset.
        
        Args:
            df (pd.DataFrame): Data to analyze
        """
        print("\nDataset Summary Statistics:")
        print("=" * 50)
        print(f"Total number of records: {len(df)}")
        print(f"Number of features: {len(df.columns) - 1}")  # Excluding target
        print()
        
        # Descriptive statistics
        stats = df[['Hours_Studied', 'Attendance', 'Final_Score']].describe()
        print("Descriptive Statistics:")
        print(stats.round(2))
        print()
        
        # Additional insights
        print("Data Insights:")
        print("-" * 20)
        
        # Hours studied insights
        avg_hours = df['Hours_Studied'].mean()
        print(f"Average study hours: {avg_hours:.1f} hours")
        
        high_performers = df[df['Final_Score'] >= 85]
        if len(high_performers) > 0:
            avg_hours_high = high_performers['Hours_Studied'].mean()
            avg_attendance_high = high_performers['Attendance'].mean()
            print(f"High performers (≥85 score) study on average: {avg_hours_high:.1f} hours")
            print(f"High performers have average attendance: {avg_attendance_high:.1f}%")
        
        low_performers = df[df['Final_Score'] < 60]
        if len(low_performers) > 0:
            avg_hours_low = low_performers['Hours_Studied'].mean()
            avg_attendance_low = low_performers['Attendance'].mean()
            print(f"Low performers (<60 score) study on average: {avg_hours_low:.1f} hours")
            print(f"Low performers have average attendance: {avg_attendance_low:.1f}%")
        
        # Missing values check
        missing_values = df.isnull().sum().sum()
        print(f"Missing values: {missing_values}")
        
        print()