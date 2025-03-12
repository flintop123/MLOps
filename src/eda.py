import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from ydata_profiling import ProfileReport

# Load the dataset
(X_train, y_train), (_, _) = fashion_mnist.load_data()
df_train = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))
df_train['label'] = y_train

# Ensure the 'reports' directory exists
os.makedirs("reports", exist_ok=True)

# Generate report
profile = ProfileReport(df, title="Fashion MNIST EDA Report")
#profile = ProfileReport(df_train, minimal=True)
#profile.to_file("fashion_mnist_eda_report.html")
#profile = ProfileReport(df_train, explorative=True, correlations={"auto": False})
report_path = "reports/fashion_mnist_eda_report.html"
profile.to_file(report_path)
print(f"EDA report saved to {report_path}")



# ========================
# VISUAL SUMMARIES
# ========================
# 1. Class Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=df)
plt.title('Class Distribution')
plt.savefig("reports/class_distribution.png")
plt.close()

# 2. Missing Values Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.title('Missing Values Heatmap')
plt.savefig("reports/missing_values.png")
plt.close()

# 3. Feature Correlation (if numerical features are available)
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation')
plt.savefig("reports/feature_correlation.png")
plt.close()

print("EDA report and visual summaries generated successfully!")



