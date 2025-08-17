"""
Interactive Web Dashboard for Student Score Prediction
Built with Streamlit for easy web-based predictions and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor
from visualizer import DataVisualizer
from model_trainer import ModelTrainer
from predictor import ScorePredictor

# Page configuration
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        color: #2e8b57;
        text-align: center;
        padding: 1rem;
        background-color: #f0f8f0;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset."""
    try:
        data_processor = DataProcessor()
        df = data_processor.load_data('data/student_data.csv')
        df_clean = data_processor.clean_data(df)
        return df_clean
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_model():
    """Load and cache the trained model."""
    try:
        # Check if model exists, if not train it
        model_path = 'models/student_score_model.pkl'
        
        if not os.path.exists(model_path):
            st.info("Training model for the first time...")
            train_model()
        
        predictor = ScorePredictor()
        predictor.load_model(model_path)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def train_model():
    """Train the model if it doesn't exist."""
    try:
        # Load data
        data_processor = DataProcessor()
        df = data_processor.load_data('data/student_data.csv')
        df_clean = data_processor.clean_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = data_processor.split_data(df_clean)
        
        # Train model
        trainer = ModelTrainer()
        model = trainer.train_model(X_train, y_train)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        trainer.save_model('models/student_score_model.pkl')
        
        st.success("Model trained successfully!")
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")

def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎓 Student Score Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data and model
    df = load_data()
    predictor = load_model()
    
    if df is None or predictor is None:
        st.error("Failed to load data or model. Please check the setup.")
        return
    
    # Sidebar for inputs
    st.sidebar.header("📊 Prediction Inputs")
    st.sidebar.markdown("Enter student information to predict final exam score:")
    
    # Input controls
    hours_studied = st.sidebar.slider(
        "Hours Studied per Day",
        min_value=0.0,
        max_value=12.0,
        value=4.0,
        step=0.5,
        help="Number of hours the student studies per day"
    )
    
    attendance = st.sidebar.slider(
        "Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0,
        help="Percentage of classes attended"
    )
    
    # Prediction button
    if st.sidebar.button("🔮 Predict Score", type="primary"):
        st.session_state.make_prediction = True
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Dataset Overview")
        
        # Dataset statistics
        st.subheader("Dataset Statistics")
        stats_df = df.describe().round(2)
        st.dataframe(stats_df, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        fig_corr = px.imshow(
            df[['Hours_Studied', 'Attendance', 'Final_Score']].corr(),
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Scatter plots
        st.subheader("Relationship Analysis")
        
        fig_scatter = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Hours Studied vs Final Score', 'Attendance vs Final Score')
        )
        
        # Hours vs Score
        fig_scatter.add_trace(
            go.Scatter(
                x=df['Hours_Studied'],
                y=df['Final_Score'],
                mode='markers',
                name='Hours vs Score',
                marker=dict(color='blue', opacity=0.6)
            ),
            row=1, col=1
        )
        
        # Attendance vs Score
        fig_scatter.add_trace(
            go.Scatter(
                x=df['Attendance'],
                y=df['Final_Score'],
                mode='markers',
                name='Attendance vs Score',
                marker=dict(color='green', opacity=0.6)
            ),
            row=1, col=2
        )
        
        fig_scatter.update_xaxes(title_text="Hours Studied", row=1, col=1)
        fig_scatter.update_xaxes(title_text="Attendance (%)", row=1, col=2)
        fig_scatter.update_yaxes(title_text="Final Score", row=1, col=1)
        fig_scatter.update_yaxes(title_text="Final Score", row=1, col=2)
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        st.header("🎯 Prediction Results")
        
        # Show current inputs
        st.subheader("Current Inputs")
        st.metric("Hours Studied", f"{hours_studied} hours")
        st.metric("Attendance", f"{attendance}%")
        
        # Make prediction
        if hasattr(st.session_state, 'make_prediction') or st.sidebar.button("Update Prediction"):
            try:
                # Get prediction with confidence
                result = predictor.predict_with_confidence(hours_studied, attendance)
                predicted_score = result['predicted_score']
                
                # Display prediction
                st.markdown(
                    f'<div class="prediction-result">Predicted Score: {predicted_score:.1f}/100</div>',
                    unsafe_allow_html=True
                )
                
                # Performance category
                if predicted_score >= 90:
                    category = "🌟 Excellent (A)"
                    color = "green"
                elif predicted_score >= 80:
                    category = "👍 Good (B)"
                    color = "blue"
                elif predicted_score >= 70:
                    category = "✅ Satisfactory (C)"
                    color = "orange"
                elif predicted_score >= 60:
                    category = "⚠️ Needs Improvement (D)"
                    color = "red"
                else:
                    category = "❌ Unsatisfactory (F)"
                    color = "darkred"
                
                st.markdown(f"**Performance Category:** :{color}[{category}]")
                
                # Score breakdown
                st.subheader("Score Breakdown")
                breakdown_data = {
                    'Component': ['Base Score', 'Hours Contribution', 'Attendance Contribution'],
                    'Points': [
                        result['base_score'],
                        result['hours_contribution'],
                        result['attendance_contribution']
                    ]
                }
                breakdown_df = pd.DataFrame(breakdown_data)
                breakdown_df['Points'] = breakdown_df['Points'].round(1)
                
                fig_breakdown = px.bar(
                    breakdown_df,
                    x='Component',
                    y='Points',
                    title='Score Contribution by Component',
                    color='Points',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_breakdown, use_container_width=True)
                
                # Recommendations
                st.subheader("📝 Recommendations")
                if hours_studied < 4:
                    st.warning("💡 Consider increasing study hours for better performance")
                if attendance < 75:
                    st.warning("💡 Improve attendance to boost final score")
                if predicted_score < 70:
                    st.error("💡 Both study hours and attendance need significant improvement")
                elif predicted_score >= 85:
                    st.success("💡 Excellent study habits! Keep up the good work")
                else:
                    st.info("💡 Good progress! Small improvements can lead to better scores")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        
        # Scenario comparison
        st.subheader("📊 Scenario Comparison")
        if st.button("Compare Scenarios"):
            scenarios = [
                (hours_studied + 2, attendance),      # More study hours
                (hours_studied, min(attendance + 15, 100)),  # Better attendance
                (hours_studied + 2, min(attendance + 15, 100)),  # Both improved
                (max(hours_studied - 1, 0), max(attendance - 20, 0))  # Both reduced
            ]
            
            scenario_names = ['Current', '+2 Hours', '+15% Attendance', 'Both Improved', 'Both Reduced']
            scenario_scores = [predictor.predict_score(hours_studied, attendance)]
            
            for hours, att in scenarios:
                score = predictor.predict_score(hours, att)
                scenario_scores.append(score)
            
            scenario_df = pd.DataFrame({
                'Scenario': scenario_names,
                'Predicted Score': scenario_scores
            })
            
            fig_scenarios = px.bar(
                scenario_df,
                x='Scenario',
                y='Predicted Score',
                title='Score Comparison Across Scenarios',
                color='Predicted Score',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_scenarios, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Student Score Prediction Dashboard** | Built with Streamlit | "
        "Predicts final exam scores based on study habits using Linear Regression"
    )

if __name__ == "__main__":
    main()