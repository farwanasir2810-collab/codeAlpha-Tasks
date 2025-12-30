"""
Credit Scoring Model - Complete Machine Learning Implementation
Includes: Data Preprocessing, Feature Engineering, Model Training, and Evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class CreditScoringModel:
    """Complete Credit Scoring ML Pipeline"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic credit data for demonstration"""
        np.random.seed(self.random_state)
        
        data = {
            'income': np.random.normal(50000, 20000, n_samples).clip(15000, 150000),
            'age': np.random.normal(40, 12, n_samples).clip(18, 70),
            'employment_length': np.random.normal(8, 5, n_samples).clip(0, 40),
            'total_debt': np.random.normal(30000, 15000, n_samples).clip(0, 100000),
            'credit_utilization': np.random.uniform(0, 100, n_samples),
            'num_credit_accounts': np.random.poisson(5, n_samples).clip(0, 20),
            'num_defaults': np.random.poisson(0.5, n_samples).clip(0, 5),
            'late_payments': np.random.poisson(1, n_samples).clip(0, 10),
            'loan_amount': np.random.normal(15000, 8000, n_samples).clip(1000, 50000),
            'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'home_ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Feature Engineering
        df['debt_to_income_ratio'] = (df['total_debt'] / df['income'] * 100).clip(0, 200)
        df['loan_to_income_ratio'] = (df['loan_amount'] / df['income'] * 100).clip(0, 100)
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], labels=['Young', 'Adult', 'Middle', 'Senior'])
        
        # Create target variable (creditworthy: 1, non-creditworthy: 0)
        # Based on logical rules
        creditworthy_score = (
            (df['income'] > 40000).astype(int) * 2 +
            (df['debt_to_income_ratio'] < 40).astype(int) * 2 +
            (df['num_defaults'] == 0).astype(int) * 2 +
            (df['credit_utilization'] < 50).astype(int) * 1 +
            (df['late_payments'] < 2).astype(int) * 1 +
            (df['employment_length'] > 2).astype(int) * 1
        )
        
        # Add some randomness
        df['creditworthy'] = (creditworthy_score >= 6).astype(int)
        noise = np.random.binomial(1, 0.1, n_samples)
        df['creditworthy'] = ((df['creditworthy'] + noise) % 2)
        
        return df
    
    def load_data(self, filepath=None):
        """Load data from file or generate sample data"""
        if filepath:
            try:
                df = pd.read_csv(filepath)
                print(f"Data loaded successfully from {filepath}")
            except:
                print("Could not load file. Generating sample data instead.")
                df = self.generate_sample_data()
        else:
            print("Generating sample credit data...")
            df = self.generate_sample_data()
        
        print(f"\nDataset Shape: {df.shape}")
        print(f"\nTarget Distribution:")
        print(df['creditworthy'].value_counts())
        print(f"\nClass Balance: {df['creditworthy'].value_counts(normalize=True)}")
        
        return df
    
    def exploratory_analysis(self, df):
        """Perform exploratory data analysis"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        print("\nDataset Info:")
        print(df.info())
        
        print("\nStatistical Summary:")
        print(df.describe())
        
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        # Correlation with target
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()['creditworthy'].sort_values(ascending=False)
        print("\nFeature Correlation with Target:")
        print(correlations)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Target distribution
        df['creditworthy'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['red', 'green'])
        axes[0, 0].set_title('Target Distribution')
        axes[0, 0].set_xlabel('Creditworthy (0=No, 1=Yes)')
        axes[0, 0].set_ylabel('Count')
        
        # Income distribution by target
        df.boxplot(column='income', by='creditworthy', ax=axes[0, 1])
        axes[0, 1].set_title('Income by Creditworthiness')
        axes[0, 1].set_xlabel('Creditworthy')
        
        # Debt to income ratio
        df.boxplot(column='debt_to_income_ratio', by='creditworthy', ax=axes[1, 0])
        axes[1, 0].set_title('Debt-to-Income Ratio by Creditworthiness')
        
        # Correlation heatmap (top features)
        top_corr_features = correlations.abs().nlargest(10).index
        sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='coolwarm', ax=axes[1, 1])
        axes[1, 1].set_title('Feature Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
        print("\nEDA visualizations saved as 'eda_analysis.png'")
        plt.show()
        
    def preprocess_data(self, df):
        """Preprocess the data"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        df_processed = df.copy()
        
        # Handle missing values
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'creditworthy']
        
        imputer = SimpleImputer(strategy='median')
        df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
        print("Missing values handled using median imputation")
        
        # Encode categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
        
        print(f"Encoded {len(categorical_cols)} categorical features")
        
        # Separate features and target
        X = df_processed.drop('creditworthy', axis=1)
        y = df_processed['creditworthy']
        
        self.feature_names = X.columns.tolist()
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.feature_names
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_names
        )
        
        print(f"\nTraining set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        print("Features scaled using StandardScaler")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def feature_importance_analysis(self):
        """Analyze feature importance using Random Forest"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Train a quick Random Forest for feature importance
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf_temp.fit(self.X_train, self.y_train)
        
        # Get feature importance
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importances.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(importances['feature'].head(10), importances['importance'].head(10))
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plot saved as 'feature_importance.png'")
        plt.show()
        
        return importances
    
    def train_models(self):
        """Train multiple classification models"""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                C=1.0
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=15,
                min_samples_split=20,
                n_jobs=-1
            )
        }
        
        # Train models
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            print(f"{name} training completed")
        
        print("\nAll models trained successfully!")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        results_list = []
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"{name}")
            print(f"{'='*50}")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='roc_auc'
            )
            
            # Store results
            results = {
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'CV ROC-AUC Mean': cv_scores.mean(),
                'CV ROC-AUC Std': cv_scores.std()
            }
            results_list.append(results)
            self.results[name] = results
            
            # Print results
            print(f"\nAccuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            print(f"\nCross-Validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Confusion Matrix
            print("\nConfusion Matrix:")
            cm = confusion_matrix(self.y_test, y_pred)
            print(cm)
            
            # Classification Report
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred, target_names=['Non-Creditworthy', 'Creditworthy']))
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_list)
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def plot_model_comparison(self, results_df):
        """Plot model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            results_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False, color='steelblue')
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_ylim([0, 1])
            ax.set_xlabel('')
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nModel comparison plot saved as 'model_comparison.png'")
        plt.show()
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("ROC curves saved as 'roc_curves.png'")
        plt.show()
    
    def predict_new_applicant(self, applicant_data):
        """Predict creditworthiness for a new applicant"""
        print("\n" + "="*60)
        print("NEW APPLICANT PREDICTION")
        print("="*60)
        
        # Use the best model (Random Forest typically performs best)
        best_model = self.models['Random Forest']
        
        # Preprocess the input
        applicant_df = pd.DataFrame([applicant_data])
        
        # Encode categorical variables
        for col, le in self.label_encoders.items():
            if col in applicant_df.columns:
                applicant_df[col] = le.transform(applicant_df[col].astype(str))
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in applicant_df.columns:
                applicant_df[feature] = 0
        
        applicant_df = applicant_df[self.feature_names]
        
        # Scale features
        applicant_scaled = self.scaler.transform(applicant_df)
        
        # Make prediction
        prediction = best_model.predict(applicant_scaled)[0]
        probability = best_model.predict_proba(applicant_scaled)[0]
        
        print("\nApplicant Information:")
        for key, value in applicant_data.items():
            print(f"  {key}: {value}")
        
        print(f"\nPrediction: {'CREDITWORTHY' if prediction == 1 else 'NON-CREDITWORTHY'}")
        print(f"Confidence: {probability[prediction] * 100:.2f}%")
        print(f"Probability of being creditworthy: {probability[1] * 100:.2f}%")
        print(f"Probability of being non-creditworthy: {probability[0] * 100:.2f}%")
        
        return prediction, probability
    
    def run_complete_pipeline(self, filepath=None):
        """Run the complete ML pipeline"""
        print("\n" + "="*60)
        print("CREDIT SCORING MODEL - COMPLETE PIPELINE")
        print("="*60)
        
        # Step 1: Load Data
        df = self.load_data(filepath)
        
        # Step 2: EDA
        self.exploratory_analysis(df)
        
        # Step 3: Preprocess
        self.preprocess_data(df)
        
        # Step 4: Feature Importance
        self.feature_importance_analysis()
        
        # Step 5: Train Models
        self.train_models()
        
        # Step 6: Evaluate Models
        results_df = self.evaluate_models()
        
        # Step 7: Visualize Results
        self.plot_model_comparison(results_df)
        self.plot_roc_curves()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated Files:")
        print("  - eda_analysis.png")
        print("  - feature_importance.png")
        print("  - model_comparison.png")
        print("  - roc_curves.png")
        
        return results_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize the model
    credit_model = CreditScoringModel(random_state=42)
    
    # Run complete pipeline
    # Option 1: Use generated sample data
    results = credit_model.run_complete_pipeline()
    
    # Option 2: Use your own data file (uncomment to use)
    # results = credit_model.run_complete_pipeline(filepath='your_data.csv')
    
    # Example: Predict for a new applicant
    print("\n" + "="*60)
    print("EXAMPLE: NEW APPLICANT PREDICTION")
    print("="*60)
    
    new_applicant = {
        'income': 60000,
        'age': 35,
        'employment_length': 5,
        'total_debt': 25000,
        'credit_utilization': 30,
        'num_credit_accounts': 4,
        'num_defaults': 0,
        'late_payments': 1,
        'loan_amount': 20000,
        'education_level': 'Bachelor',
        'home_ownership': 'Own',
        'debt_to_income_ratio': 41.67,
        'loan_to_income_ratio': 33.33,
        'age_group': 'Adult'
    }
    
    prediction, probability = credit_model.predict_new_applicant(new_applicant)
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)