"""
=================================================================
DISEASE PREDICTION FROM MEDICAL DATA - MACHINE LEARNING PROJECT
=================================================================
Author: ML Student Project
Objective: Predict disease possibility using patient medical data
Dataset: Heart Disease Dataset (UCI ML Repository)
"""

# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("DISEASE PREDICTION SYSTEM - MACHINE LEARNING PROJECT")
print("=" * 70)

# ============================================================
# STEP 2: LOAD AND EXPLORE THE DATASET
# ============================================================
print("\n[STEP 1] Loading Dataset...")

# For demonstration, creating sample heart disease data
# In real project, use: df = pd.read_csv('heart.csv')
np.random.seed(42)
n_samples = 500

data = {
    'age': np.random.randint(25, 80, n_samples),
    'sex': np.random.randint(0, 2, n_samples),
    'chest_pain': np.random.randint(0, 4, n_samples),
    'resting_bp': np.random.randint(90, 200, n_samples),
    'cholesterol': np.random.randint(120, 400, n_samples),
    'fasting_bs': np.random.randint(0, 2, n_samples),
    'resting_ecg': np.random.randint(0, 3, n_samples),
    'max_heart_rate': np.random.randint(70, 200, n_samples),
    'exercise_angina': np.random.randint(0, 2, n_samples),
    'oldpeak': np.random.uniform(0, 6, n_samples),
    'st_slope': np.random.randint(0, 3, n_samples),
}

# Create target (disease: 0=No, 1=Yes) with some logic
disease = ((data['age'] > 55) & (data['cholesterol'] > 240) & 
           (data['chest_pain'] >= 2)).astype(int)
data['disease'] = disease

df = pd.DataFrame(data)

print(f"✓ Dataset loaded successfully!")
print(f"  Total samples: {len(df)}")
print(f"  Total features: {df.shape[1] - 1}")
print(f"\n  First 5 rows:")
print(df.head())

# ============================================================
# STEP 3: DATA EXPLORATION AND ANALYSIS
# ============================================================
print("\n[STEP 2] Exploring Dataset...")

print("\n  Dataset Information:")
print(df.info())

print("\n  Statistical Summary:")
print(df.describe())

print("\n  Disease Distribution:")
print(df['disease'].value_counts())
print(f"  - No Disease: {(df['disease']==0).sum()} ({(df['disease']==0).sum()/len(df)*100:.1f}%)")
print(f"  - Disease: {(df['disease']==1).sum()} ({(df['disease']==1).sum()/len(df)*100:.1f}%)")

# Check for missing values
print("\n  Missing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values found!")
else:
    print(missing[missing > 0])

# ============================================================
# STEP 4: DATA PREPROCESSING
# ============================================================
print("\n[STEP 3] Preprocessing Data...")

# Separate features and target
X = df.drop('disease', axis=1)
y = df['disease']

print(f"  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Training set size: {len(X_train)} samples")
print(f"  Testing set size: {len(X_test)} samples")

# Feature Scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("  ✓ Data scaled using StandardScaler")

# ============================================================
# STEP 5: BUILD AND TRAIN MODELS
# ============================================================
print("\n[STEP 4] Training Machine Learning Models...")
print("-" * 70)

# Initialize models
models = {
    'Support Vector Machine (SVM)': SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n  Training {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"    ✓ Accuracy: {accuracy:.4f}")
    print(f"    ✓ Precision: {precision:.4f}")
    print(f"    ✓ Recall: {recall:.4f}")
    print(f"    ✓ F1-Score: {f1:.4f}")

# ============================================================
# STEP 6: MODEL COMPARISON
# ============================================================
print("\n[STEP 5] Comparing Model Performance...")
print("=" * 70)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1_score'] for m in results.keys()]
})

print("\n  Performance Comparison Table:")
print(comparison_df.to_string(index=False))

# Find best model
best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
best_accuracy = comparison_df['Accuracy'].max()

print(f"\n  🏆 BEST MODEL: {best_model_name}")
print(f"  🎯 Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# ============================================================
# STEP 7: VISUALIZATIONS
# ============================================================
print("\n[STEP 6] Creating Visualizations...")

# 1. Model Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Accuracy comparison
axes[0, 0].bar(comparison_df['Model'], comparison_df['Accuracy'], color='skyblue')
axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_ylim([0, 1])
axes[0, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(comparison_df['Accuracy']):
    axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center')

# All metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(comparison_df['Model']))
width = 0.2

axes[0, 1].bar(x - 1.5*width, comparison_df['Accuracy'], width, label='Accuracy', color='skyblue')
axes[0, 1].bar(x - 0.5*width, comparison_df['Precision'], width, label='Precision', color='lightgreen')
axes[0, 1].bar(x + 0.5*width, comparison_df['Recall'], width, label='Recall', color='lightcoral')
axes[0, 1].bar(x + 1.5*width, comparison_df['F1-Score'], width, label='F1-Score', color='gold')
axes[0, 1].set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
axes[0, 1].legend()
axes[0, 1].set_ylim([0, 1])

# 2. Confusion Matrix for Best Model
cm = results[best_model_name]['confusion_matrix']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xlabel('Predicted')

# 3. Feature Importance (for Random Forest)
if 'Random Forest' in results:
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    axes[1, 1].barh(feature_importance['feature'], feature_importance['importance'], color='coral')
    axes[1, 1].set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Visualizations saved as 'model_comparison.png'")

# ============================================================
# STEP 8: PREDICTION FUNCTION
# ============================================================
print("\n[STEP 7] Creating Prediction Function...")

def predict_disease(age, sex, chest_pain, resting_bp, cholesterol, 
                    fasting_bs, resting_ecg, max_heart_rate, 
                    exercise_angina, oldpeak, st_slope):
    """
    Predict disease based on patient information
    
    Parameters:
    - age: Age of patient
    - sex: Gender (0=Female, 1=Male)
    - chest_pain: Type of chest pain (0-3)
    - resting_bp: Resting blood pressure
    - cholesterol: Serum cholesterol
    - fasting_bs: Fasting blood sugar (0 or 1)
    - resting_ecg: Resting ECG results (0-2)
    - max_heart_rate: Maximum heart rate achieved
    - exercise_angina: Exercise induced angina (0 or 1)
    - oldpeak: ST depression
    - st_slope: Slope of peak exercise ST segment (0-2)
    
    Returns:
    - Prediction and probability
    """
    
    # Create input array
    input_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol,
                           fasting_bs, resting_ecg, max_heart_rate,
                           exercise_angina, oldpeak, st_slope]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Use best model for prediction
    best_model = results[best_model_name]['model']
    prediction = best_model.predict(input_scaled)[0]
    
    # Get probability if available
    if hasattr(best_model, 'predict_proba'):
        probability = best_model.predict_proba(input_scaled)[0]
        return prediction, probability
    else:
        return prediction, None

# Test the prediction function
print("\n  Testing prediction function with sample patient...")
sample_patient = {
    'age': 60,
    'sex': 1,
    'chest_pain': 3,
    'resting_bp': 145,
    'cholesterol': 280,
    'fasting_bs': 1,
    'resting_ecg': 1,
    'max_heart_rate': 120,
    'exercise_angina': 1,
    'oldpeak': 2.5,
    'st_slope': 2
}

prediction, probability = predict_disease(**sample_patient)

print(f"\n  Sample Patient Data:")
for key, value in sample_patient.items():
    print(f"    {key}: {value}")

print(f"\n  Prediction Result:")
if prediction == 1:
    print(f"    ⚠️  Disease Detected: YES")
else:
    print(f"    ✓ Disease Detected: NO")

if probability is not None:
    print(f"    Probability: No Disease={probability[0]:.2%}, Disease={probability[1]:.2%}")

# ============================================================
# STEP 9: FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(f"""
✓ Dataset: Heart Disease Prediction
✓ Total Samples: {len(df)}
✓ Features: {X.shape[1]}
✓ Models Trained: {len(models)}
✓ Best Model: {best_model_name}
✓ Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)

Models Performance:
""")

for name in results.keys():
    print(f"  • {name}:")
    print(f"      Accuracy: {results[name]['accuracy']:.4f}")
    print(f"      F1-Score: {results[name]['f1_score']:.4f}")

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 70)

# Save the best model (optional)
# import joblib
# joblib.dump(results[best_model_name]['model'], 'best_disease_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# print("\n✓ Best model and scaler saved!")