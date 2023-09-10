import umap
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# Function definition to create a CNN model
def create_cnn_model(N, n_features, n_classes):
    model = Sequential()
    model.add(Conv1D(N, kernel_size=3, activation='relu', input_shape=(n_features, 1)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(2*N, kernel_size=3, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(4*N, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model


# Load the training data
train_data = pd.read_csv("train_data_swc.csv")
# Extract the features (X) and target labels (y) from the training data
X = train_data.drop("y", axis=1)
y = train_data["y"]

# Load the test data
X_test = pd.read_csv("test_data_swc.csv")

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

# Standardize the data using the training data's statistics
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

n_classes = len(y.unique())     # Number of classes: 9
assert n_classes == 9
n_features = X_train.shape[1]   # Number of features: 108
assert n_features == 108
model_filename = "stacking_model.pkl"   # Define the filename for saving the model

# One-hot encode the target labels
# y_train -= 1
# y_val -= 1
# y_train = to_categorical(y_train, num_classes=n_classes)
# y_val = to_categorical(y_val, num_classes=n_classes)

# Define the parameter grid for hyperparameter tuning
params_grid_pipeline = {
    "umap__n_components": [15, 25, 50],
    "umap__n_neighbors": [5, 10, 20],
    "umap__min_dist": [0.05, 0.1, 0.2],
    'cnn__N': [32, 64, 128],
    'cnn__epochs': [6, 10],
    'cnn__batch_size': [32, 64],
}

# Create a KerasClassifier with UMAP-transformed features
cnn_model = KerasClassifier(build_fn=create_cnn_model, verbose=0)
pipeline = Pipeline([("umap", umap.UMAP()), ("cnn", cnn_model)])
grid_search = GridSearchCV(estimator=pipeline, param_grid=params_grid_pipeline, cv=3, verbose=2, n_jobs=-1)

# Fit the GridSearchCV to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Access the best hyperparameters
best_params = grid_search.best_params_

# Apply UMAP to reduce dimensionality
umap_model = umap.UMAP(n_components=best_params['umap__n_components'],
                       n_neighbors=best_params['umap__n_neighbors'], min_dist=best_params['umap__min_dist'])
X_train = umap_model.fit_transform(X_train)
X_val = umap_model.transform(X_val)
X_test = umap_model.transform(X_test)

# Create the final CNN model with UMAP-transformed features
final_model = create_cnn_model(best_params['cnn__N'], best_params['umap__n_components'], n_classes)

# Train the final model on the UMAP-transformed data
final_model.fit(X_train, y_train, epochs=best_params['cnn__epochs'], batch_size=best_params['cnn__batch_size'])

# Create base models (CNNs) as scikit-learn estimators
# base_models = [
#     ("cnn1", KerasClassifier(build_fn=lambda: create_cnn_model(best_params['N'], best_params['n_components'], n_classes),
#                              epochs=best_params['epochs'], batch_size=best_params['batch_size']))
# ]

# Define the stacking ensemble model
# final_model = StackingClassifier(estimators=base_models, final_estimator=LGBMClassifier(n_estimators=30, n_jobs=-1, force_col_wise=True))

# Fit the optimized model on the scaled training data
# final_model.fit(X_train, y_train)

# Save the optimized model to a file
# joblib.dump(final_model, model_filename)

# Make predictions on the scaled validation set using the optimized model
val_predictions = final_model.predict(X_val)

# Calculate prediction probabilities for validation predictions
val_proba = final_model.predict_proba(X_val)

# Clip predicted probabilities to avoid extremes of the log function
val_proba = np.clip(val_proba, a_min=1e-15, a_max=1 - 1e-15)

# Calculate log loss for validation predictions
val_log_loss = log_loss(y_val, val_proba)
print(f"Validation Log Loss: {val_log_loss:.4f}")

# Make predictions on the scaled test data using the optimized model
test_predictions = final_model.predict(X_test)

# Calculate prediction probabilities for test predictions
test_proba = final_model.predict_proba(X_test)

# Clip predicted probabilities to avoid extremes of the log function
test_proba = np.clip(test_proba, a_min=1e-15, a_max=1 - 1e-15)

# Save the test predictions to a CSV file
submission_df = pd.DataFrame(test_proba, columns=[f"c{i}" for i in range(1, n_classes + 1)])
submission_df.to_csv("test_predictions.csv", index=False)
