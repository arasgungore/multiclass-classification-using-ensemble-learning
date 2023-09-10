import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import log_loss
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

# Create base models (CNNs) as scikit-learn estimators
base_models = [
    ("cnn1", KerasClassifier(build_fn=lambda: create_cnn_model(256, n_features, n_classes), epochs=15, batch_size=32))
]

# Define the stacking ensemble model
stacking_model = StackingClassifier(estimators=base_models, final_estimator=LGBMClassifier(n_estimators=30, n_jobs=-1, force_col_wise=True))

# Fit the stacking model on the scaled training data
stacking_model.fit(X_train, y_train)

# Load the saved stacking model from a file
# loaded_stacking_model = joblib.load(model_filename)

# Save the trained stacking model to a file
# joblib.dump(stacking_model, model_filename)

# Make predictions on the scaled validation set
val_predictions = stacking_model.predict(X_val)

# Calculate prediction probabilities for validation predictions
val_proba = stacking_model.predict_proba(X_val)

# Clip predicted probabilities to avoid extremes of the log function
val_proba = np.clip(val_proba, a_min=1e-15, a_max=1 - 1e-15)

# Calculate log loss for validation predictions
val_log_loss = log_loss(y_val, val_proba)
print(f"Validation Log Loss: {val_log_loss:.4f}")

# Make predictions on the scaled test data
test_predictions = stacking_model.predict(X_test)

# Calculate prediction probabilities for test predictions
test_proba = stacking_model.predict_proba(X_test)

# Clip predicted probabilities to avoid extremes of the log function
test_proba = np.clip(test_proba, a_min=1e-15, a_max=1 - 1e-15)

# Save the test predictions to a CSV file
submission_df = pd.DataFrame(test_proba, columns=[f"c{i}" for i in range(1, n_classes + 1)])
submission_df.to_csv("test_predictions.csv", index=False)
