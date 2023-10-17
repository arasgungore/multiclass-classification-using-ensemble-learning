import umap
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import StackingClassifier
from skopt import BayesSearchCV

# Load the training data
train_data = pd.read_csv("./swc-dataset/train_data_swc.csv")
# Extract the features (X) and target labels (y) from the training data
X = train_data.drop("y", axis=1)
y = train_data["y"]

# Load the test data
X_test = pd.read_csv("./swc-dataset/test_data_swc.csv")

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standardize the data using the training data's statistics
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

n_classes = len(y.unique())     # Number of classes: 9
assert n_classes == 9
n_features = X_train.shape[1]   # Number of features: 108
assert n_features == 108

# Define LightGBM classifier
lgb_model = LGBMClassifier()

# Define Decision Tree classifier
dt_model = DecisionTreeClassifier()

# Stack the models using Logistic Regression as the second-level model
stacked_model = StackingClassifier(
    estimators=[('lgb', lgb_model), ('decision_tree', dt_model)],
    final_estimator=LGBMClassifier()
)

# Define the parameter space for Bayesian optimization
param_space = {
    'lgb__n_estimators': (10, 100, 400),
    'lgb__learning_rate': (0.1, 0.2),
    'lgb__num_leaves': (30, 60, 100),
    'lgb__max_depth': (5, 20),
    'decision_tree__max_depth': (5, 20),
}

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform hyperparameter tuning with Bayesian optimization
opt = BayesSearchCV(
    stacked_model, param_space, cv=cv, n_iter=50, scoring='neg_log_loss', n_jobs=-1, random_state=42
)
opt.fit(X_train, y_train)

# Get the best hyperparameters
best_params = opt.best_params_

# Train the final model with the best hyperparameters
final_model = stacked_model.set_params(**best_params)
final_model.fit(X_train, y_train)

# Make predictions on the validation set
val_predictions = final_model.predict(X_val)

# Calculate prediction probabilities for validation predictions
val_proba = final_model.predict_proba(X_val)

# Clip predicted probabilities to avoid extremes of the log function
val_proba = np.clip(val_proba, a_min=1e-15, a_max=1 - 1e-15)

# Calculate log loss for validation predictions
val_log_loss = log_loss(y_val, val_proba)
print(f"Validation Log Loss: {val_log_loss:.4f}")

# Make predictions on the test data
test_predictions = final_model.predict(X_test)

# Calculate prediction probabilities for test predictions
test_proba = final_model.predict_proba(X_test)

# Clip predicted probabilities to avoid extremes of the log function
test_proba = np.clip(test_proba, a_min=1e-15, a_max=1 - 1e-15)

# Create a DataFrame for test predictions
submission_df = pd.DataFrame(test_proba, columns=[f"c{i}" for i in range(1, n_classes + 1)])

# Save the test predictions to a CSV file
submission_df.to_csv("test_predictions.csv", index=False)
