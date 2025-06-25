import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STUDENT EXAM SCORE PREDICTION - MODEL TRAINING")
print("=" * 60)

# Set visual style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("\n📊 Loading and preparing data...")
# Load dataset
try:
    data = pd.read_csv("student_performance_50000.csv")
    print(f"✅ Loaded {len(data)} records from CSV")
except FileNotFoundError:
    print("📝 CSV not found, creating comprehensive sample data...")
    np.random.seed(42)
    n_samples = 10000
    
    # Create more realistic sample data
    hours_studied = np.random.gamma(2, 2, n_samples)  # Gamma distribution for study hours
    hours_studied = np.clip(hours_studied, 0.5, 12)  # Clip to realistic range
    
    previous_score = np.random.normal(75, 15, n_samples)  # Normal distribution
    previous_score = np.clip(previous_score, 30, 100)
    
    attendance = np.random.beta(8, 2, n_samples) * 40 + 60  # Beta distribution shifted
    attendance = np.clip(attendance, 50, 100)
    
    # Generate realistic exam scores with some noise
    exam_score = (
        hours_studied * 3.5 +  # Study hours impact
        previous_score * 0.65 +  # Previous performance impact
        attendance * 0.25 +  # Attendance impact
        np.random.normal(0, 6, n_samples)  # Random noise
    )
    exam_score = np.clip(exam_score, 0, 100)
    
    data = pd.DataFrame({
        'Hours_Studied': hours_studied,
        'Previous_Score': previous_score,
        'Attendance': attendance,
        'Exam_Score': exam_score
    })
    
    print(f"✅ Created {len(data)} realistic sample records")

# Display basic statistics
print(f"\n📈 Dataset Overview:")
print(f"   • Total Records: {len(data)}")
print(f"   • Features: {', '.join(data.columns[:-1])}")
print(f"   • Target: {data.columns[-1]}")
print(f"\n📊 Data Statistics:")
print(data.describe().round(2))

# Feature selection
X = data[['Hours_Studied', 'Previous_Score', 'Attendance']]
y = data['Exam_Score']

print(f"\n🔄 Splitting data (80% train, 20% test)...")
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   • Training samples: {len(X_train)}")
print(f"   • Testing samples: {len(X_test)}")

print(f"\n🤖 Training multiple regression models...")
# Define comprehensive set of models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Support Vector Regression": SVR(kernel='rbf', C=100, gamma=0.1),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
}

# Train and evaluate models
results = []
model_objects = {}

for name, model in models.items():
    print(f"   🔄 Training {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, train_preds)
    test_mse = mean_squared_error(y_test, test_preds)
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results.append({
        "Model": name,
        "Train_MSE": train_mse,
        "Test_MSE": test_mse,
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        "Test_MAE": test_mae,
        "CV_R2_Mean": cv_mean,
        "CV_R2_Std": cv_std,
        "Overfitting": train_r2 - test_r2
    })
    
    model_objects[name] = model

# Create comparison DataFrame
comparison_df = pd.DataFrame(results).sort_values(by="Test_R2", ascending=False)

print(f"\n📊 MODEL COMPARISON RESULTS:")
print("=" * 80)
display_cols = ['Model', 'Test_R2', 'Test_MSE', 'Test_MAE', 'CV_R2_Mean', 'Overfitting']
print(comparison_df[display_cols].round(4).to_string(index=False))

# Select best model
best_model_entry = comparison_df.iloc[0]
best_model = model_objects[best_model_entry['Model']]
best_model_name = best_model_entry['Model']

print(f"\n🏆 BEST MODEL SELECTED: {best_model_name}")
print(f"   • Test R² Score: {best_model_entry['Test_R2']:.4f}")
print(f"   • Test MSE: {best_model_entry['Test_MSE']:.4f}")
print(f"   • Cross-validation R²: {best_model_entry['CV_R2_Mean']:.4f} (±{best_model_entry['CV_R2_Std']:.4f})")

# Save best model and comparison results
print(f"\n💾 Saving model and results...")
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model_comparison.pkl", "wb") as f:
    pickle.dump(comparison_df, f)

print("✅ Model and comparison results saved!")

# Test prediction
print(f"\n🔮 Testing prediction...")
new_student_data = pd.DataFrame([{
    'Hours_Studied': 6,
    'Previous_Score': 70,
    'Attendance': 90
}])
predicted_score = best_model.predict(new_student_data)[0]
print(f"   Sample prediction (6 hours, 70 previous, 90% attendance): {predicted_score:.2f}")

# Generate comprehensive PDF report
print(f"\n📄 Generating comprehensive PDF report...")
with PdfPages("student_exam_analysis_report.pdf") as pdf:
    
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
    best_preds = best_model.predict(X_test)
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
    from scipy import stats
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
        from scipy.stats import gaussian_kde
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
        
        data[f'{feature}_Category'] = pd.cut(data[feature], bins=bins, labels=labels, include_lowest=True)
        
        # Box plot
        box_data = [data[data[f'{feature}_Category'] == cat]['Exam_Score'].dropna() 
                   for cat in labels if not data[data[f'{feature}_Category'] == cat]['Exam_Score'].empty]
        valid_labels = [label for i, label in enumerate(labels) 
                       if not data[data[f'{feature}_Category'] == label]['Exam_Score'].empty]
        
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

print("✅ Comprehensive PDF report generated: 'student_exam_analysis_report.pdf'")
print("\n🎉 Training completed successfully!")
print("=" * 60)
print("Next steps:")
print("1. Run 'streamlit run app.py' to start the prediction app")
print("2. Check the PDF report for detailed analysis")
print("=" * 60)