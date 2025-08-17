"""
Simple script to run the ML pipeline and train the model.
Run this first before using the dashboard.
"""

import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor
from model_trainer import ModelTrainer

def run_pipeline():
    """Run the complete ML pipeline."""
    print("🎓 Student Score Prediction - Training Pipeline")
    print("=" * 50)
    
    try:
        # Initialize components
        data_processor = DataProcessor()
        trainer = ModelTrainer()
        
        # File paths
        data_path = 'data/student_data.csv'
        model_path = 'models/student_score_model.pkl'
        
        print("\n1. Loading and processing data...")
        df = data_processor.load_data(data_path)
        df_clean = data_processor.clean_data(df)
        
        print("\n2. Splitting data...")
        X_train, X_test, y_train, y_test = data_processor.split_data(df_clean)
        
        print("\n3. Training model...")
        model = trainer.train_model(X_train, y_train)
        
        print("\n4. Evaluating model...")
        metrics = trainer.evaluate_model(X_test, y_test)
        
        print("\n5. Saving model...")
        os.makedirs('models', exist_ok=True)
        trainer.save_model(model_path)
        
        print("\n✅ Pipeline completed successfully!")
        print(f"📊 Model R² Score: {metrics['r2_score']:.3f}")
        print(f"📊 Mean Absolute Error: {metrics['mean_absolute_error']:.2f} points")
        print(f"💾 Model saved to: {model_path}")
        
        print("\n🚀 You can now run the dashboard with:")
        print("   streamlit run dashboard.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_pipeline()
    if not success:
        sys.exit(1)