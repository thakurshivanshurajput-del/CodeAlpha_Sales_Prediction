"""
Sales Prediction Using Multiple Linear Regression
Author: Shivanshu
Project for CodeAlpha Data Science Internship
Task: Sales Prediction using Python
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set plotting aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

def load_and_prepare_data(filepath):
    """Loads advertising dataset and prepares features."""
    print("--> Loading Advertising dataset...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset '{filepath}' not found in the current directory.")
        
    df = pd.read_csv(filepath)
    
    # Drop unnecessary index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    print(f"--> Data loaded successfully. Shape: {df.shape}")
    print(f"--> Missing values check: {df.isnull().sum().sum()} missing values found.")
    return df

def train_and_evaluate_model(df):
    """Performs feature analysis, splits data, trains regression model, and saves plots."""
    print("\n--> Starting exploratory analysis and feature selection...")
    
    # Create directory for saving visualizations
    os.makedirs('sales_plots', exist_ok=True)
    
    # 1. Feature Correlation Analysis
    corr_matrix = df.corr()
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title('Correlation Matrix of Advertising Channels', fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('sales_plots/correlation_matrix.png', dpi=300)
    plt.close()
    
    # 2. Train-Test Split (80% Train, 20% Test)
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Model Predictions & Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # 5. Visualizing Actual vs Predicted Sales
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=y_pred, color='#e63946', alpha=0.8, edgecolors='w', s=60)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Fit')
    plt.title('Actual vs. Predicted Sales Performance', fontweight='bold', pad=12)
    plt.xlabel('Actual Sales (in Thousands)')
    plt.ylabel('Predicted Sales (in Thousands)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('sales_plots/actual_vs_predicted.png', dpi=300)
    plt.close()
    
    # 6. Visualizing Feature Impact (Coefficients)
    coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    
    plt.figure(figsize=(6, 4))
    sns.barplot(x=coefficients.index, y=coefficients['Coefficient'], palette='viridis')
    plt.title('Impact of Ad Spend Channels on Sales Outcomes', fontweight='bold', pad=12)
    plt.ylabel('Sales Return Value per Unit Spend')
    plt.xlabel('Advertising Medium')
    plt.tight_layout()
    plt.savefig('sales_plots/feature_coefficients.png', dpi=300)
    plt.close()
    
    # --- PRINT METRICS TO CONSOLE ---
    print("\n" + "="*50)
    print("         SALES FORECASTING MODEL METRICS        ")
    print("="*50)
    print(f"• Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"• Variance Explained (R² Score) : {r2:.4f} (~{r2*100:.1f}%)")
    print("-"*50)
    print("Model Parameters (Impact per $1,000 spend):")
    print(f"  - Base Intercept Baseline : {model.intercept_:.4f}")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  - {feature:<9} Coefficient   : {coef:.4f}")
    print("="*50)
    print("--> All analytical plots exported successfully inside './sales_plots/' folder.")

if __name__ == "__main__":
    DATA_FILE = "Advertising.csv"
    try:
        data = load_and_prepare_data(DATA_FILE)
        train_and_evaluate_model(data)
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")