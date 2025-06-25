# Student Exam Score Predictor App - Enhanced Version with PDF Download

import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
from scipy import stats
import base64

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
    .download-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .download-button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border: none;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem;
        transition: transform 0.2s;
    }
    .download-button:hover {
        transform: translateY(-2px);
        text-decoration: none;
        color: white;
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
        ["🔮 Prediction", "📈 Model Comparison", "📄 Analysis Report", "📊 PDF Downloads", "ℹ️ About"]
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

# Function to generate PDF report
@st.cache_data
def generate_pdf_report(data, _best_model, comparison_df, best_model_name, best_model_entry):
    """Generate comprehensive PDF report"""
    buffer = BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Set visual style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Split data for predictions
        X = data[['Hours_Studied', 'Previous_Score', 'Attendance']]
        y = data['Exam_Score']
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. Model Comparison Bar Chart
        plt.figure(figsize=(12, 8))
        model_comparison = comparison_df.head(6)  # Top 6 models
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_comparison)))
        
        bars = plt.barh(model_comparison['Model'], model_comparison['Test_R2'], color=colors)
        plt.xlabel('R² Score (Test Set)', fontsize=12)
        plt.ylabel('Regression Models', fontsize=12)
        plt.title('Model Performance Comparison\n(Higher R² Score = Better Performance)', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, r2) in enumerate(zip(bars, model_comparison['Test_R2'])):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{r2:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        
        # 2. Detailed Model Metrics Comparison
        plt.figure(figsize=(14, 8))
        metrics_df = comparison_df.head(6)[['Model', 'Test_R2', 'Test_MSE', 'Test_MAE']].set_index('Model')
        
        # Normalize metrics for better visualization
        normalized_metrics = metrics_df.copy()
        normalized_metrics['Test_MSE'] = 1 - (normalized_metrics['Test_MSE'] / normalized_metrics['Test_MSE'].max())
        normalized_metrics['Test_MAE'] = 1 - (normalized_metrics['Test_MAE'] / normalized_metrics['Test_MAE'].max())
        
        x = np.arange(len(normalized_metrics.index))
        width = 0.25
        
        plt.bar(x - width, normalized_metrics['Test_R2'], width, label='R² Score', alpha=0.8)
        plt.bar(x, normalized_metrics['Test_MSE'], width, label='MSE (Inverted)', alpha=0.8)
        plt.bar(x + width, normalized_metrics['Test_MAE'], width, label='MAE (Inverted)', alpha=0.8)
        
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Normalized Score (Higher = Better)', fontsize=12)
        plt.title('Comprehensive Model Performance Metrics', fontsize=14, fontweight='bold')
        plt.xticks(x, normalized_metrics.index, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        
        # 3. Best Model - Actual vs Predicted
        best_preds = _best_model.predict(X_test)
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, best_preds, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_score, max_score = y_test.min(), y_test.max()
        plt.plot([min_score, max_score], [min_score, max_score], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Exam Score', fontsize=12)
        plt.ylabel('Predicted Exam Score', fontsize=12)
        plt.title(f'Actual vs Predicted Exam Scores\n{best_model_name} (R² = {best_model_entry["Test_R2"]:.4f})', 
                  fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = np.corrcoef(y_test, best_preds)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}', transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=11)
        
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        
        # 4. Feature Correlation Heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = data.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.3f')
        plt.title('Feature Correlation Matrix\n(Understanding Relationships Between Variables)', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        
        # 5. Residual Analysis
        residuals = y_test - best_preds
        plt.figure(figsize=(12, 8))
        
        # Residual plot
        plt.subplot(2, 2, 1)
        plt.scatter(best_preds, residuals, alpha=0.6, s=30)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        # Histogram of residuals
        plt.subplot(2, 2, 2)
        plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        plt.grid(True, alpha=0.3)
        
        # Q-Q plot
        plt.subplot(2, 2, 3)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Normality Check)')
        plt.grid(True, alpha=0.3)
        
        # Residuals vs Features
        plt.subplot(2, 2, 4)
        plt.scatter(y_test, residuals, alpha=0.6, s=30)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Actual Values')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Residual Analysis - {best_model_name}', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        
        # 6. Feature Distribution Analysis
        plt.figure(figsize=(15, 10))
        
        features = ['Hours_Studied', 'Previous_Score', 'Attendance', 'Exam_Score']
        for i, feature in enumerate(features, 1):
            plt.subplot(2, 2, i)
            
            # Histogram with KDE
            plt.hist(data[feature], bins=30, alpha=0.7, color='lightblue', edgecolor='black', density=True)
            
            # Add KDE curve
            kde = gaussian_kde(data[feature])
            x_range = np.linspace(data[feature].min(), data[feature].max(), 100)
            plt.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            plt.xlabel(feature.replace('_', ' '))
            plt.ylabel('Density')
            plt.title(f'Distribution of {feature.replace("_", " ")}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.suptitle('Feature Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        
        # 7. Feature Impact Analysis
        plt.figure(figsize=(15, 10))
        
        features = ['Hours_Studied', 'Previous_Score', 'Attendance']
        for i, feature in enumerate(features, 1):
            plt.subplot(2, 3, i)
            plt.scatter(data[feature], data['Exam_Score'], alpha=0.5, s=20)
            
            # Add regression line
            z = np.polyfit(data[feature], data['Exam_Score'], 1)
            p = np.poly1d(z)
            plt.plot(data[feature], p(data[feature]), "r--", linewidth=2)
            
            # Calculate correlation
            corr = data[feature].corr(data['Exam_Score'])
            plt.xlabel(feature.replace('_', ' '))
            plt.ylabel('Exam Score')
            plt.title(f'{feature.replace("_", " ")} vs Exam Score\n(Correlation: {corr:.3f})')
            plt.grid(True, alpha=0.3)
            
            # Box plot for categorical analysis
            plt.subplot(2, 3, i + 3)
            # Create bins for continuous variables
            if feature == 'Hours_Studied':
                bins = [0, 3, 6, 9, 12]
                labels = ['Low (0-3h)', 'Medium (3-6h)', 'High (6-9h)', 'Very High (9-12h)']
            elif feature == 'Previous_Score':
                bins = [0, 60, 75, 85, 100]
                labels = ['Poor (<60)', 'Average (60-75)', 'Good (75-85)', 'Excellent (85+)']
            else:  # Attendance
                bins = [50, 70, 85, 95, 100]
                labels = ['Poor (<70%)', 'Average (70-85%)', 'Good (85-95%)', 'Excellent (95%+)']
            
            data_temp = data.copy()
            data_temp[f'{feature}_Category'] = pd.cut(data_temp[feature], bins=bins, labels=labels, include_lowest=True)
            
            # Box plot
            box_data = [data_temp[data_temp[f'{feature}_Category'] == cat]['Exam_Score'].dropna() 
                       for cat in labels if not data_temp[data_temp[f'{feature}_Category'] == cat]['Exam_Score'].empty]
            valid_labels = [label for i, label in enumerate(labels) 
                           if not data_temp[data_temp[f'{feature}_Category'] == label]['Exam_Score'].empty]
            
            if box_data:
                plt.boxplot(box_data, labels=valid_labels)
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Exam Score')
                plt.title(f'Score Distribution by {feature.replace("_", " ")} Level')
                plt.grid(True, alpha=0.3)
        
        plt.suptitle('Feature Impact on Exam Scores', fontsize=16, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()
        
        # 8. Summary Statistics Page
        plt.figure(figsize=(12, 10))
        plt.axis('off')
        
        # Create text summary
        summary_text = f"""
        STUDENT EXAM SCORE PREDICTION - ANALYSIS SUMMARY
        
        Dataset Overview:
        • Total Students: {len(data):,}
        • Average Exam Score: {data['Exam_Score'].mean():.2f}
        • Score Standard Deviation: {data['Exam_Score'].std():.2f}
        • Score Range: {data['Exam_Score'].min():.1f} - {data['Exam_Score'].max():.1f}
        
        Best Performing Model:
        • Model Type: {best_model_name}
        • Test R² Score: {best_model_entry['Test_R2']:.4f}
        • Test MSE: {best_model_entry['Test_MSE']:.2f}
        • Test MAE: {best_model_entry['Test_MAE']:.2f}
        • Cross-Validation R² Mean: {best_model_entry['CV_R2_Mean']:.4f}
        
        Feature Correlations with Exam Score:
        • Hours Studied: {data['Hours_Studied'].corr(data['Exam_Score']):.3f}
        • Previous Score: {data['Previous_Score'].corr(data['Exam_Score']):.3f}
        • Attendance: {data['Attendance'].corr(data['Exam_Score']):.3f}
        
        Key Insights:
        • Previous academic performance is the strongest predictor
        • Study hours show moderate positive correlation with exam scores
        • Attendance has a positive but weaker correlation
        • The best model explains {best_model_entry['Test_R2']*100:.1f}% of score variance
        
        Model Performance Ranking (Top 5):
        """
        
        for i, row in comparison_df.head(5).iterrows():
            summary_text += f"        {i+1}. {row['Model']} (R² = {row['Test_R2']:.4f})\n"
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        plt.title('Analysis Summary Report', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(bbox_inches='tight')
        plt.close()
    
    buffer.seek(0)
    return buffer

# Function to create download link
def create_download_link(buffer, filename, text):
    """Create a download link for the PDF"""
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" class="download-button">{text}</a>'
    return href

best_model, comparison_df = load_models()
data = load_data()

if best_model is None:
    st.error("❌ Failed to load models. Please retrain the models.")
    st.stop()

# Get best model info
best_model_name = comparison_df.iloc[0]['Model']
best_r2 = comparison_df.iloc[0]['Test_R2']
best_model_entry = comparison_df.iloc[0]

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
        # Dataset Overview
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", f"{len(data):,}")
        
        with col2:
            st.metric("Average Score", f"{data['Exam_Score'].mean():.1f}")
        
        with col3:
            st.metric("Score Std Dev", f"{data['Exam_Score'].std():.1f}")
        
        with col4:
            st.metric("Score Range", f"{data['Exam_Score'].min():.0f}-{data['Exam_Score'].max():.0f}")
        
        # Feature Statistics
        st.subheader("📈 Feature Statistics")
        
        stats_df = data.describe().round(2)
        st.dataframe(stats_df, use_container_width=True)
        
        # Correlation Analysis
        st.subheader("🔗 Feature Correlations")
        
        corr_matrix = data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
        plt.close()
        
        # Feature Distributions
        st.subheader("📊 Feature Distributions")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        features = ['Hours_Studied', 'Previous_Score', 'Attendance', 'Exam_Score']
        
        for i, feature in enumerate(features):
            ax = axes[i//2, i%2]
            ax.hist(data[feature], bins=30, alpha=0.7, edgecolor='black')
            ax.set_title(f'Distribution of {feature.replace("_", " ")}')
            ax.set_xlabel(feature.replace("_", " "))
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Best Model Performance
        st.subheader("🏆 Best Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="feature-importance">
                <h4>🎯 Best Model: {best_model_name}</h4>
                <p><strong>R² Score:</strong> {best_r2:.4f}</p>
                <p><strong>MSE:</strong> {best_model_entry['Test_MSE']:.2f}</p>
                <p><strong>MAE:</strong> {best_model_entry['Test_MAE']:.2f}</p>
                <p><strong>CV R² Mean:</strong> {best_model_entry['CV_R2_Mean']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Feature importance visualization
            feature_names = ['Hours_Studied', 'Previous_Score', 'Attendance']
            correlations = [data[feature].corr(data['Exam_Score']) for feature in feature_names]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(feature_names, correlations, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_title('Feature Correlation with Exam Score')
            ax.set_ylabel('Correlation Coefficient')
            ax.set_xticklabels([name.replace('_', ' ') for name in feature_names], rotation=45)
            
            # Add value labels on bars
            for bar, corr in zip(bars, correlations):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    else:
        st.error("No data available for analysis.")

elif page == "📊 PDF Downloads":
    st.header("📊 Download Analysis Reports")
    
    st.markdown("""
    <div class="download-section">
        <h3>📄 Available Reports</h3>
        <p>Generate and download comprehensive analysis reports in PDF format.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if data is not None and best_model is not None:
        # Generate PDF Report
        st.subheader("📈 Comprehensive Analysis Report")
        
        if st.button("🔄 Generate PDF Report", type="primary"):
            with st.spinner("Generating comprehensive PDF report... This may take a moment."):
                try:
                    # Generate the PDF report
                    pdf_buffer = generate_pdf_report(data, best_model, comparison_df, best_model_name, best_model_entry)
                    
                    # Create download link
                    st.success("✅ PDF Report Generated Successfully!")
                    
                    # Display download button
                    st.download_button(
                        label="📥 Download Comprehensive Analysis Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"student_score_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
                    
                    # Report contents preview
                    st.subheader("📋 Report Contents")
                    st.info("""
                    **This comprehensive report includes:**
                    
                    1. **Model Performance Comparison** - Bar charts comparing all models
                    2. **Detailed Metrics Analysis** - R², MSE, MAE comparisons
                    3. **Actual vs Predicted Visualization** - Scatter plot with correlation
                    4. **Feature Correlation Matrix** - Heatmap showing relationships
                    5. **Residual Analysis** - Model diagnostic plots
                    6. **Feature Distribution Analysis** - Histograms with KDE curves
                    7. **Feature Impact Analysis** - Scatter plots and box plots
                    8. **Summary Statistics** - Complete analysis summary
                    """)
                    
                except Exception as e:
                    st.error(f"Error generating PDF report: {str(e)}")
        
        # Individual Chart Downloads
        st.subheader("📊 Individual Chart Downloads")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Download Model Comparison Chart"):
                # Generate individual model comparison chart
                fig, ax = plt.subplots(figsize=(12, 8))
                top_models = comparison_df.head(6)
                colors = plt.cm.Set3(np.linspace(0, 1, len(top_models)))
                
                bars = ax.barh(top_models['Model'], top_models['Test_R2'], color=colors)
                ax.set_xlabel('R² Score')
                ax.set_title('Model Performance Comparison')
                ax.grid(axis='x', alpha=0.3)
                
                for bar, r2 in zip(bars, top_models['Test_R2']):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{r2:.3f}', ha='left', va='center', fontweight='bold')
                
                # Save to buffer
                buf = BytesIO()
                plt.savefig(buf, format='pdf', bbox_inches='tight')
                buf.seek(0)
                plt.close()
                
                st.download_button(
                    label="📥 Download Model Comparison PDF",
                    data=buf.getvalue(),
                    file_name=f"model_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        
        with col2:
            if st.button("🔗 Download Correlation Matrix"):
                # Generate correlation matrix
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = data.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax)
                ax.set_title('Feature Correlation Matrix')
                
                # Save to buffer
                buf = BytesIO()
                plt.savefig(buf, format='pdf', bbox_inches='tight')
                buf.seek(0)
                plt.close()
                
                st.download_button(
                    label="📥 Download Correlation Matrix PDF",
                    data=buf.getvalue(),
                    file_name=f"correlation_matrix_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        
        # Data Export Options
        st.subheader("💾 Data Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = data.to_csv(index=False)
            st.download_button(
                label="📊 Download Raw Data (CSV)",
                data=csv_data,
                file_name=f"student_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            model_comparison_csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="🏆 Download Model Results (CSV)",
                data=model_comparison_csv,
                file_name=f"model_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            # Create summary statistics CSV
            summary_stats = data.describe()
            summary_csv = summary_stats.to_csv()
            st.download_button(
                label="📈 Download Summary Stats (CSV)",
                data=summary_csv,
                file_name=f"summary_statistics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
    else:
        st.error("❌ Data or model not available. Please ensure the training process completed successfully.")

elif page == "ℹ️ About":
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## 🎓 Student Exam Score Predictor
    
    This application uses machine learning to predict student exam scores based on three key factors:
    
    ### 📊 Input Features
    - **📚 Hours Studied**: Daily study hours (0-12 hours)
    - **📈 Previous Score**: Previous academic performance (0-100)
    - **🎯 Attendance**: Class attendance percentage (50-100%)
    
    ### 🤖 Machine Learning Models
    Our system evaluates multiple regression models:
    - **Random Forest Regressor**
    - **Gradient Boosting Regressor**
    - **Support Vector Regressor**
    - **Linear Regression**
    - **Ridge Regression**
    - **Lasso Regression**
    - **Decision Tree Regressor**
    - **K-Nearest Neighbors**
    - **AdaBoost Regressor**
    - **Extra Trees Regressor**
    
    ### 🎯 Key Features
    - **Real-time Predictions**: Get instant score predictions
    - **Model Comparison**: Compare performance across different algorithms
    - **Comprehensive Analysis**: Detailed statistical analysis and visualizations
    - **PDF Reports**: Download professional analysis reports
    - **Data Export**: Export data and results in multiple formats
    
    ### 📈 Model Evaluation Metrics
    - **R² Score**: Measures how well the model explains variance
    - **Mean Squared Error (MSE)**: Average squared prediction errors
    - **Mean Absolute Error (MAE)**: Average absolute prediction errors
    - **Cross-Validation**: Ensures model robustness
    
    ### 🔍 How It Works
    1. **Data Collection**: Gather student performance data
    2. **Feature Engineering**: Process and prepare input features
    3. **Model Training**: Train multiple ML algorithms
    4. **Model Selection**: Choose the best performing model
    5. **Prediction**: Make predictions on new student data
    6. **Analysis**: Generate comprehensive reports and insights
    
    ### 💡 Use Cases
    - **Academic Planning**: Help students plan study schedules
    - **Early Intervention**: Identify students who need additional support
    - **Performance Tracking**: Monitor academic progress over time
    - **Resource Allocation**: Optimize educational resource distribution
    
    ### 🚀 Technical Stack
    - **Framework**: Streamlit for web interface
    - **ML Libraries**: Scikit-learn for machine learning
    - **Visualization**: Matplotlib, Seaborn for charts
    - **Data Processing**: Pandas, NumPy for data manipulation
    - **PDF Generation**: Matplotlib backends for report generation
    
    ### 📝 Model Assumptions
    - Linear relationships between features and exam scores
    - Independent observations
    - Normal distribution of residuals
    - Homoscedasticity (constant variance)
    
    ### ⚠️ Limitations
    - Predictions are based on historical patterns
    - External factors not captured in the model may affect actual performance
    - Model accuracy depends on data quality and quantity
    - Should be used as a supplementary tool, not the sole decision-making factor
    
    ### 📞 Support
    For technical support or questions about the application, please refer to the documentation or contact the development team.
    
    ---
    
    **Version**: 2.0 Enhanced  
    **Last Updated**: June 2025  
    **Developed with**: Python, Streamlit, Scikit-learn
    """)
    
    # System Information
    st.subheader("🔧 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Dataset Size**: {len(data):,} students  
        **Features**: 3 input features  
        **Target**: Exam Score (0-100)  
        **Best Model**: {best_model_name}  
        **Model Accuracy**: {best_r2:.1%}
        """)
    
    with col2:
        st.info(f"""
        **Models Evaluated**: {len(comparison_df)} algorithms  
        **Validation Method**: 5-Fold Cross-Validation  
        **Scoring Metric**: R² Score  
        **Best R² Score**: {best_r2:.4f}  
        **Best MSE**: {best_model_entry['Test_MSE']:.2f}
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🎓 Student Exam Score Predictor | Built with ❤️ using Streamlit</p>
    <p>Empowering education through data-driven insights</p>
</div>
""", unsafe_allow_html=True)
