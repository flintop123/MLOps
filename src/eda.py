import pandas as pd
from tensorflow.keras.datasets import fashion_mnist
from ydata_profiling import ProfileReport

# Load the dataset
(X_train, y_train), (_, _) = fashion_mnist.load_data()
df_train = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))
df_train['label'] = y_train

# Generate report
profile = ProfileReport(df_train, minimal=True)
#profile.to_file("fashion_mnist_eda_report.html")
#profile = ProfileReport(df_train, explorative=True, correlations={"auto": False})
report_path = "reports/fashion_mnist_eda_report.html"
profile.to_file(report_path)
print(f"EDA report saved to {report_path}")
