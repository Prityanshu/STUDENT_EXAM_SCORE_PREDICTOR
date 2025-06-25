# Student Exam Score Predictor App - Enhanced Version

import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .stAlert > div {
        padding: 1rem;
    }
    .feature-importance {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎓 Student Exam Score Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 Navigation")
    page = st.selectbox(
        "Choose a section:",
        ["🔮 Prediction", "📈 Model Comparison", "📄 Analysis Report", "ℹ️ About"]
    )

# Check if files exist
if not os.path.exists("best_model.pkl"):
    st.error("❌ Model file not found!")
    st.info("Please run `python train_model.py` first to create the model.")
    st.stop()

# Load models and data
@st.cache_resource
def load_models():
    try:
        with open("best_model.pkl", "rb") as f:
            best_model = pickle.load(f)
        
        with open("model_comparison.pkl", "rb") as f:
            comparison_df = pickle.load(f)
        
        return best_model, comparison_df
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Replace the load_data function in your app.py with this updated version

@st.cache_data
def load_data():
    try:
        # Try multiple possible file names
        possible_files = [
            "student_performance_50000.csv",  # From training script
            "student_scores.csv",             # Original app expectation
            "student_performance.csv"         # Alternative name
        ]
        
        for filename in possible_files:
            if os.path.exists(filename):
                print(f"Loading data from: {filename}")
                return pd.read_csv(filename)
        
        # If no file found, generate sample data
        print("No CSV file found, generating sample data...")
        np.random.seed(42)
        n_samples = 1000
        hours = np.random.uniform(1, 10, n_samples)
        previous = np.random.uniform(50, 95, n_samples)
        attendance = np.random.uniform(60, 100, n_samples)
        
        # Create realistic exam scores with some noise
        exam_score = (
            0.3 * hours * 10 +
            0.5 * previous +
            0.2 * attendance +
            np.random.normal(0, 5, n_samples)
        )
        exam_score = np.clip(exam_score, 0, 100)
        
        df = pd.DataFrame({
            'Hours_Studied': hours,
            'Previous_Score': previous,
            'Attendance': attendance,
            'Exam_Score': exam_score
        })
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

best_model, comparison_df = load_models()
data = load_data()

if best_model is None:
    st.error("❌ Failed to load models. Please retrain the models.")
    st.stop()

# Get best model info
best_model_name = comparison_df.iloc[0]['Model']
best_r2 = comparison_df.iloc[0]['Test_R2']

# Main content based on page selection
if page == "🔮 Prediction":
    st.header("🔮 Predict Student Exam Score")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Enter Student Information")
        
        # Input form
        with st.form("prediction_form"):
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                hours = st.number_input(
                    "📚 Hours Studied",
                    min_value=0.0,
                    max_value=12.0,
                    value=6.0,
                    step=0.5,
                    help="Number of hours studied per day"
                )
            
            with col_b:
                previous = st.number_input(
                    "📊 Previous Score",
                    min_value=0.0,
                    max_value=100.0,
                    value=75.0,
                    step=1.0,
                    help="Previous exam score (0-100)"
                )
            
            with col_c:
                attendance = st.number_input(
                    "🎯 Attendance (%)",
                    min_value=50.0,
                    max_value=100.0,
                    value=85.0,
                    step=1.0,
                    help="Attendance percentage"
                )
            
            submitted = st.form_submit_button("🔮 Predict Score", type="primary")
        
        if submitted:
            # Create input data
            input_data = pd.DataFrame([{
                'Hours_Studied': hours,
                'Previous_Score': previous,
                'Attendance': attendance
            }])
            
            # Make prediction
            try:
                predicted_score = best_model.predict(input_data)[0]
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-result">
                    <h2>📊 Predicted Exam Score</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">{predicted_score:.1f}</h1>
                    <p>Model: {best_model_name} (R² = {best_r2:.3f})</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Performance interpretation
                if predicted_score >= 90:
                    st.success("🎉 **Excellent Performance Expected!** This student is likely to perform exceptionally well.")
                    st.balloons()
                elif predicted_score >= 80:
                    st.success("👍 **Good Performance Expected!** This student should perform well above average.")
                elif predicted_score >= 70:
                    st.warning("📚 **Average Performance Expected.** Consider additional study time or support.")
                elif predicted_score >= 60:
                    st.warning("⚠️ **Below Average Performance.** This student may need significant additional support.")
                else:
                    st.error("🚨 **Poor Performance Predicted.** Immediate intervention and support recommended.")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                
                recommendations = []
                if hours < 4:
                    recommendations.append("📚 **Increase study time** - Current study hours are below optimal level")
                if previous < 70:
                    recommendations.append("🎯 **Focus on fundamentals** - Previous performance indicates need for foundational work")
                if attendance < 80:
                    recommendations.append("🏫 **Improve attendance** - Regular class attendance is crucial for success")
                
                if recommendations:
                    for rec in recommendations:
                        st.info(rec)
                else:
                    st.success("✅ **Current study pattern looks good!** Continue with the same approach.")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with col2:
        st.subheader("📊 Input Summary")
        
        # Create a nice summary card
        st.markdown(f"""
        <div class="metric-container">
            <h4>📚 Study Hours</h4>
            <h2>{hours}</h2>
            <p>Hours per day</p>
        </div>
        
        <div class="metric-container">
            <h4>📊 Previous Score</h4>
            <h2>{previous}</h2>
            <p>Out of 100</p>
        </div>
        
        <div class="metric-container">
            <h4>🎯 Attendance</h4>
            <h2>{attendance}%</h2>
            <p>Class attendance</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        st.subheader("📈 Quick Stats")
        if hours >= 6:
            st.success("✅ Good study habits")
        else:
            st.warning("⚠️ Consider more study time")
        
        if previous >= 75:
            st.success("✅ Strong academic foundation")
        else:
            st.info("💡 Room for improvement")
        
        if attendance >= 85:
            st.success("✅ Excellent attendance")
        else:
            st.warning("⚠️ Attendance needs improvement")

elif page == "📈 Model Comparison":
    st.header("📈 Model Performance Comparison")
    
    # Display comparison table
    st.subheader("🏆 Model Rankings")
    
    # Format the comparison dataframe for display
    display_df = comparison_df.copy()
    display_df['Test_R2'] = display_df['Test_R2'].round(4)
    display_df['Test_MSE'] = display_df['Test_MSE'].round(2)
    display_df['Test_MAE'] = display_df['Test_MAE'].round(2)
    display_df['CV_R2_Mean'] = display_df['CV_R2_Mean'].round(4)
    display_df['Overfitting'] = display_df['Overfitting'].round(4)
    
    # Add rank column
    display_df['Rank'] = range(1, len(display_df) + 1)
    
    # Reorder columns
    cols = ['Rank', 'Model', 'Test_R2', 'Test_MSE', 'Test_MAE', 'CV_R2_Mean', 'Overfitting']
    st.dataframe(
        display_df[cols],
        use_container_width=True,
        hide_index=True
    )
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 R² Score Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get top 6 models
        top_models = comparison_df.head(6)
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_models)))
        
        bars = ax.barh(top_models['Model'], top_models['Test_R2'], color=colors)
        ax.set_xlabel('R² Score')
        ax.set_title('Model Performance (Test R² Score)')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, r2 in zip(bars, top_models['Test_R2']):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                   f'{r2:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("📈 Mean Squared Error Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(top_models['Model'], top_models['Test_MSE'], color=colors)
        ax.set_xlabel('Mean Squared Error')
        ax.set_title('Model Performance (Test MSE)')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, mse in zip(bars, top_models['Test_MSE']):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{mse:.2f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Model details
    st.subheader("🔍 Model Details")
    selected_model = st.selectbox("Select a model to view details:", comparison_df['Model'].tolist())
    
    model_info = comparison_df[comparison_df['Model'] == selected_model].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test R² Score", f"{model_info['Test_R2']:.4f}")
    
    with col2:
        st.metric("Test MSE", f"{model_info['Test_MSE']:.2f}")
    
    with col3:
        st.metric("Test MAE", f"{model_info['Test_MAE']:.2f}")
    
    with col4:
        st.metric("CV R² Mean", f"{model_info['CV_R2_Mean']:.4f}")

elif page == "📄 Analysis Report":
    st.header("📄 Data Analysis Report")
    
    if data is not None:
        # Dataset overview
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", len(data))
        
        with col2:
            st.metric("Average Score", f"{data['Exam_Score'].mean():.1f}")
        
        with col3:
            st.metric("Score Range", f"{data['Exam_Score'].max() - data['Exam_Score'].min():.1f}")
        
        with col4:
            st.metric("Std Deviation", f"{data['Exam_Score'].std():.1f}")
        
        # Data distribution
        st.subheader("📈 Score Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(data['Exam_Score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(data['Exam_Score'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {data["Exam_Score"].mean():.1f}')
            ax.set_xlabel('Exam Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Exam Scores')
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot([data['Hours_Studied'], data['Previous_Score'], data['Attendance']], 
                      labels=['Hours Studied', 'Previous Score', 'Attendance'])
            ax.set_title('Feature Distributions')
            ax.set_ylabel('Value')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        
        corr_matrix = data.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, fmt='.3f')
        ax.set_title('Feature Correlation Heatmap')
        st.pyplot(fig)
        plt.close()
        
        # Feature relationships
        st.subheader("📊 Feature Relationships with Exam Score")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(data['Hours_Studied'], data['Exam_Score'], alpha=0.6, color='blue')
            ax.set_xlabel('Hours Studied')
            ax.set_ylabel('Exam Score')
            ax.set_title('Hours Studied vs Exam Score')
            ax.grid(alpha=0.3)
            
            # Add trend line
            z = np.polyfit(data['Hours_Studied'], data['Exam_Score'], 1)
            p = np.poly1d(z)
            ax.plot(data['Hours_Studied'], p(data['Hours_Studied']), "r--", alpha=0.8)
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(data['Previous_Score'], data['Exam_Score'], alpha=0.6, color='green')
            ax.set_xlabel('Previous Score')
            ax.set_ylabel('Exam Score')
            ax.set_title('Previous Score vs Exam Score')
            ax.grid(alpha=0.3)
            
            # Add trend line
            z = np.polyfit(data['Previous_Score'], data['Exam_Score'], 1)
            p = np.poly1d(z)
            ax.plot(data['Previous_Score'], p(data['Previous_Score']), "r--", alpha=0.8)
            
            st.pyplot(fig)
            plt.close()
        
        with col3:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(data['Attendance'], data['Exam_Score'], alpha=0.6, color='orange')
            ax.set_xlabel('Attendance (%)')
            ax.set_ylabel('Exam Score')
            ax.set_title('Attendance vs Exam Score')
            ax.grid(alpha=0.3)
            
            # Add trend line
            z = np.polyfit(data['Attendance'], data['Exam_Score'], 1)
            p = np.poly1d(z)
            ax.plot(data['Attendance'], p(data['Attendance']), "r--", alpha=0.8)
            
            st.pyplot(fig)
            plt.close()
        
        # Statistical summary
        st.subheader("📋 Statistical Summary")
        st.dataframe(data.describe().round(2), use_container_width=True)
        
        # Feature importance (if available)
        st.subheader("🎯 Feature Importance")
        
        try:
            if hasattr(best_model, 'feature_importances_'):
                feature_names = ['Hours_Studied', 'Previous_Score', 'Attendance']
                importances = best_model.feature_importances_
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(feature_names, importances, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                ax.set_xlabel('Features')
                ax.set_ylabel('Importance')
                ax.set_title('Feature Importance')
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar, imp in zip(bars, importances):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{imp:.3f}', ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig)
                plt.close()
                
                # Feature importance insights
                max_feature_idx = np.argmax(importances)
                max_feature = feature_names[max_feature_idx]
                
                st.markdown(f"""
                <div class="feature-importance">
                    <h4>🔍 Key Insight</h4>
                    <p><strong>{max_feature}</strong> is the most important feature for predicting exam scores 
                    with an importance score of <strong>{importances[max_feature_idx]:.3f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
            elif hasattr(best_model, 'coef_'):
                feature_names = ['Hours_Studied', 'Previous_Score', 'Attendance']
                coefficients = best_model.coef_
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['red' if coef < 0 else 'green' for coef in coefficients]
                bars = ax.bar(feature_names, coefficients, color=colors, alpha=0.7)
                ax.set_xlabel('Features')
                ax.set_ylabel('Coefficient Value')
                ax.set_title('Linear Model Coefficients')
                ax.grid(axis='y', alpha=0.3)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Add value labels
                for bar, coef in zip(bars, coefficients):
                    ax.text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + (0.5 if coef > 0 else -0.5), 
                           f'{coef:.2f}', ha='center', va='bottom' if coef > 0 else 'top', 
                           fontweight='bold')
                
                st.pyplot(fig)
                plt.close()
                
                # Coefficient insights
                max_coef_idx = np.argmax(np.abs(coefficients))
                max_feature = feature_names[max_coef_idx]
                
                st.markdown(f"""
                <div class="feature-importance">
                    <h4>🔍 Key Insight</h4>
                    <p><strong>{max_feature}</strong> has the strongest linear relationship with exam scores 
                    with a coefficient of <strong>{coefficients[max_coef_idx]:.3f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.info("Feature importance analysis not available for this model type.")

elif page == "ℹ️ About":
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## 🎯 Purpose
    This application predicts student exam scores based on three key factors:
    - **Hours Studied**: Daily study time
    - **Previous Score**: Historical academic performance
    - **Attendance**: Class attendance percentage
    
    ## 🤖 Machine Learning Models
    The app trains and compares multiple machine learning models:
    - **Linear Regression**: Simple linear relationship modeling
    - **Random Forest**: Ensemble method with decision trees
    - **Gradient Boosting**: Sequential learning with boosting
    - **Support Vector Regression**: Support vector machine for regression
    - **Decision Tree**: Single decision tree model
    - **Ridge Regression**: Linear regression with L2 regularization
    - **Lasso Regression**: Linear regression with L1 regularization
    - **Elastic Net**: Linear regression with combined L1/L2 regularization
    
    ## 📊 Features
    - **Real-time Predictions**: Get instant score predictions
    - **Model Comparison**: Compare performance of different algorithms
    - **Data Analysis**: Comprehensive statistical analysis
    - **Interactive Visualizations**: Charts and graphs for better understanding
    - **Smart Recommendations**: Personalized study advice
    
    ## 🛠️ Technologies Used
    - **Streamlit**: Web application framework
    - **Scikit-learn**: Machine learning library
    - **Pandas**: Data manipulation and analysis
    - **Matplotlib/Seaborn**: Data visualization
    - **NumPy**: Numerical computations
    
    ## 📈 Model Performance
    The application automatically selects the best-performing model based on:
    - **R² Score**: Coefficient of determination
    - **Mean Squared Error**: Average squared differences
    - **Cross-validation**: Robust performance evaluation
    - **Overfitting Detection**: Generalization capability
    
    ## 🎓 Educational Value
    This tool helps:
    - **Students**: Understand factors affecting academic performance
    - **Educators**: Identify students who may need additional support
    - **Researchers**: Analyze educational data patterns
    - **Parents**: Track and improve study habits
    
    ## 🔒 Privacy & Data
    - No personal data is stored permanently
    - All computations are performed locally
    - Predictions are based on statistical patterns, not individual identification
    
    ## 📞 Support
    For questions or suggestions about this application, please refer to the documentation
    or contact the development team.
    
    ---
    
    ### 🏆 Model Selection Criteria
    The best model is selected based on:
    1. **Highest Test R² Score**: Better explanatory power
    2. **Lowest Cross-validation Error**: Better generalization
    3. **Minimal Overfitting**: Stable performance on new data
    4. **Balanced Performance**: Good trade-off between bias and variance
    
    ### 📊 Performance Metrics Explained
    - **R² Score**: Proportion of variance explained (0-1, higher is better)
    - **MSE**: Mean Squared Error (lower is better)
    - **MAE**: Mean Absolute Error (lower is better)
    - **CV Score**: Cross-validation R² (consistency measure)
    - **Overfitting**: Training R² - Test R² (lower is better)
    """)
    
    # Add some fun facts
    st.subheader("🎉 Fun Facts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📚 Study Insight**
        
        Research shows that consistent daily study of 2-3 hours 
        is more effective than cramming for long hours before exams!
        """)
    
    with col2:
        st.info("""
        **🎯 Attendance Impact**
        
        Students with 90%+ attendance typically score 15-20% 
        higher than those with poor attendance records.
        """)
    
    # Display current model info
    if best_model is not None:
        st.subheader("🤖 Current Best Model")
        st.success(f"""
        **Model**: {best_model_name}  
        **R² Score**: {best_r2:.4f}  
        **Performance**: {'Excellent' if best_r2 > 0.9 else 'Good' if best_r2 > 0.8 else 'Fair'}
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🎓 Student Exam Score Predictor | Built with Streamlit & Machine Learning</p>
    <p>© 2024 Educational Analytics | Version 2.0</p>
</div>
""", unsafe_allow_html=True)