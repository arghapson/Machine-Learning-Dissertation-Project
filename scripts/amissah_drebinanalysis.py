import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from google.colab import drive
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Mount Google Drive
drive.mount('/content/drive')

# Load the dataset
file_path = '/content/drive/MyDrive/AI project/datasets/Drebin-dataset.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Information:")
print(data.info())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Viewing the different types of values in column 92
column_92_values = data.iloc[:, 92].value_counts()
print(column_92_values)

# Replace '?' with NaN
data = data.replace('?', np.NaN)

# Display total missing values
total_missing_values = sum(data.isna().sum())
print("Total missing values: ", total_missing_values)

# Display dataset
display(data)

# Removing rows with missing values
data = data.dropna()

# Display information about the new dataset
print("Shape of data:", data.shape)
data

# Plot class distribution using data with categorical variables to distinguish between benign and malicious
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=data)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Initialise the LabelEncoder
label_encoder = LabelEncoder()

# Label encode the target class
data['class'] = label_encoder.fit_transform(data['class'])

# Convert all columns to numeric
for column in data.columns:
    data[column] = pd.to_numeric(data[column])

# Display the updated dataset
print("Label Encoded Dataset:")
display(data)

# Plot class distribution using encoded data
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=data)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Class distribution using data
print("\nClass Distribution:")
print(data['class'].value_counts())

# Display the updated dataset
display(data)

# Standardize the data
scaler = StandardScaler()
data_scaled = data.copy()
data_scaled.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

# Display the scaled dataset
print("Scaled Dataset:")
display(data_scaled)

"""FEATURE SELECTION"""

# Correlation Analysis with Target Variable
correlation_with_target = data.corrwith(data['class']).sort_values(ascending=False)

# Convert correlation with target variable to DataFrame
corr_with_target_df = pd.DataFrame(correlation_with_target, columns=['Correlation'])

# Save correlation with target variable to an Excel file
excel_file_path = '/content/drive/MyDrive/AI project/datasets/Correlation_With_Target.xlsx' # Define the file path
corr_with_target_df.to_excel(excel_file_path, index=True) # Save correlation to Excel
print("Correlation with Target Variable saved to:", excel_file_path)

'''# Remove variables with low correlation (threshold of 0.3)
threshold = 0.3
high_corr_features = corr_with_target_df[abs(corr_with_target_df['Correlation']) >= threshold].index.tolist()

# Count the number of features removed
num_features_removed = data.shape[1] - len(high_corr_features)
print(f"Number of features removed due to low correlation: {num_features_removed}")

# Keep only high correlation features in the dataset
data_high_corr = data[high_corr_features]

# Updated information about the dataset
print("Shape of data after removing low correlation features:", data_high_corr.shape)'''

'''# Function to train and evaluate the Random Forest model
def train_evaluate_rf(X, y, description):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Oversample the minority class using SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Initialize and train the Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_res, y_train_res)

    # Predict on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nRandom Forest Model Performance ({description}):")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

'''# Prepare data for modeling
X_all_features = data_scaled.drop(columns='class')
y = data_scaled['class']

# Train and evaluate the model with all features
train_evaluate_rf(X_all_features, y, "All Features")

# Prepare data with high correlation features
X_high_corr_features = data_high_corr.drop(columns='class')

# Train and evaluate the model with high correlation features
train_evaluate_rf(X_high_corr_features, y, "High Correlation Features")

# Calculate correlation matrix
corr_matrix = round(data.corr(), 2)

# Convert correlation matrix to DataFrame
corr_df = pd.DataFrame(corr_matrix)

# Save correlation matrix to an Excel file
excel_file_path = '/content/drive/MyDrive/AI project/datasets/Correlation_Matrix.xlsx' # Define the file path
corr_df.to_excel(excel_file_path, index=True) # Save correlation matrix to Excel
print("Correlation Matrix saved to:", excel_file_path)

# Define the threshold for high correlation
threshold = 0.7

# Find variables with high correlation
high_corr_vars = set()  # Set to store variable pairs with high correlation
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname_i = corr_matrix.columns[i]
            colname_j = corr_matrix.columns[j]
            high_corr_vars.add((colname_i, colname_j))

# Print the number of highly correlated variable pairs
num_removed_features = len(high_corr_vars)

# Remove one variable from each pair of highly correlated variables
for pair in high_corr_vars:
    # Check if variables still exist in the dataset before dropping
    if pair[0] in data_scaled.columns and pair[1] in data_scaled.columns:
        # Compute the mean correlation of each variable in the pair with the target variable
        corr_with_target_i = abs(data_scaled[pair[0]].corr(data_scaled['class']))
        corr_with_target_j = abs(data_scaled[pair[1]].corr(data_scaled['class']))

        # Remove the variable with lower absolute correlation with the target variable
        if corr_with_target_i > corr_with_target_j:
            data_scaled.drop(pair[1], axis=1, inplace=True)
        else:
            data_scaled.drop(pair[0], axis=1, inplace=True)

# Print the number of features removed and the shape of the updated dataset
print("Number of features removed due to high correlation:", num_removed_features)
print("Shape of the updated dataset after removing highly correlated variables:", data_scaled.shape)

# Define X and y before removing highly correlated features
X_before = data_scaled.drop('class', axis=1)
y_before = data_scaled['class']

# Split the dataset into train and test sets before removing highly correlated features
X_train_before, X_test_before, y_train_before, y_test_before = train_test_split(X_before, y_before, test_size=0.2, random_state=42)

# Train Random Forest model before removing highly correlated features
rfc_before = RandomForestClassifier(random_state=42)
rfc_before.fit(X_train_before, y_train_before)

# Predictions before removing highly correlated features
y_pred_before = rfc_before.predict(X_test_before)

# Calculate accuracy before removing highly correlated features
accuracy_before = accuracy_score(y_test_before, y_pred_before)

print("Accuracy before removing highly correlated features:", accuracy_before)

# Define X and y after removing highly correlated features
X_after = data_scaled.drop('class', axis=1)
y_after = data_scaled['class']

# Convert to NumPy arrays if they are DataFrames
if isinstance(X_after, pd.DataFrame):
    X_after = X_after.values
if isinstance(y_after, pd.Series):
    y_after = y_after.values

# Split the dataset into train and test sets after removing highly correlated features
X_train_after, X_test_after, y_train_after, y_test_after = train_test_split(X_after, y_after, test_size=0.2, random_state=42)

# Train Random Forest model after removing highly correlated features
rfc_after = RandomForestClassifier(random_state=42)
rfc_after.fit(X_train_after, y_train_after)

# Predictions after removing highly correlated features
y_pred_after = rfc_after.predict(X_test_after)

# Calculate accuracy after removing highly correlated features
accuracy_after = accuracy_score(y_test_after, y_pred_after)

print("Accuracy after removing highly correlated features:", accuracy_after)

"""Random Forest Feature Selection"""

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_after, y_train_after)

# Get feature importances and sort them
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Evaluate model performance for different numbers of top features
from sklearn.model_selection import cross_val_score

num_features_list = list(range(1, X_train_after.shape[1] + 1))
mean_scores = []

for num_features in num_features_list:
    selected_indices = indices[:num_features]
    X_train_selected = X_train_after[:, selected_indices]
    scores = cross_val_score(rf, X_train_selected, y_train_after, cv=5, scoring='accuracy')
    mean_scores.append(np.mean(scores))

# Find the optimal number of features
optimal_num_features = num_features_list[np.argmax(mean_scores)]

print(f'Optimal number of features: {optimal_num_features}')

# Select the top features
selected_indices = indices[:optimal_num_features]
X_train_selected = X_train_after[:, selected_indices]
X_test_selected = X_test_after[:, selected_indices]

from sklearn.ensemble import RandomForestClassifier

def select_features_rf(X_train, y_train, X_test, n_features):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    return X_train[:, indices], X_test[:, indices]

# Optimal number of features
optimal_num_features = 137

# Perform feature selection
X_train_rf, X_test_rf = select_features_rf(X_train_after, y_train_after, X_test_after, optimal_num_features)

"""Data Balancing"""

# Balancing methods
# Create copies of the training data for resampling methods
X_train1, y_train1 = X_train_after.copy(), y_train_after.copy()
X_train2, y_train2 = X_train_after.copy(), y_train_after.copy()
X_train3, y_train3 = X_train_after.copy(), y_train_after.copy()

# Finding best parameters for SMOTE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Define the pipeline
pipeline_smote = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define the parameter grid
param_grid_smote = {
    'smote__sampling_strategy': [0.5, 0.75, 1.0],
    'smote__k_neighbors': [3, 5, 7],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30]
}

# Initialize Grid Search
grid_search_smote = GridSearchCV(pipeline_smote, param_grid_smote, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model
grid_search_smote.fit(X_train_before, y_train_before)

# Display the best parameters and best score
print("Best parameters for SMOTE:", grid_search_smote.best_params_)
print("Best score for SMOTE:", grid_search_smote.best_score_)

from imblearn.over_sampling import ADASYN

# Define the pipeline
pipeline_adasyn = Pipeline([
    ('adasyn', ADASYN(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define the parameter grid
param_grid_adasyn = {
    'adasyn__sampling_strategy': [0.5, 0.75, 1.0],
    'adasyn__n_neighbors': [3, 5, 7],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30]
}

# Initialize Grid Search
grid_search_adasyn = GridSearchCV(pipeline_adasyn, param_grid_adasyn, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model
grid_search_adasyn.fit(X_train_before, y_train_before)

# Display the best parameters and best score
print("Best parameters for ADASYN:", grid_search_adasyn.best_params_)
print("Best score for ADASYN:", grid_search_adasyn.best_score_)

from imblearn.under_sampling import RandomUnderSampler

# Define the pipeline
pipeline_rus = Pipeline([
    ('rus', RandomUnderSampler(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define the parameter grid
param_grid_rus = {
    'rus__sampling_strategy': ['auto', 0.5, 0.75],
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30]
}

# Initialize Grid Search
grid_search_rus = GridSearchCV(pipeline_rus, param_grid_rus, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model
grid_search_rus.fit(X_train_before, y_train_before)

# Display the best parameters and best score
print("Best parameters for Random Under Sampling:", grid_search_rus.best_params_)
print("Best score for Random Under Sampling:", grid_search_rus.best_score_)

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Define the parameters
smote_params = {'k_neighbors': 5, 'sampling_strategy': 0.75}
classifier_params = {'max_depth': 30, 'n_estimators': 300}

# Initialize the classifier
classifier = RandomForestClassifier(random_state=42, **classifier_params)

# Initialize SMOTE with specified parameters
smote = SMOTE(random_state=42, **smote_params)

# Apply SMOTE to the training data
X_train_smote, y_train_smote = smote.fit_resample(X_train1, y_train1)

from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier

# Initialize ADASYN with specified parameters
adasyn = ADASYN(n_neighbors=5, sampling_strategy=1.0, random_state=42)

# Generate synthetic samples
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train2, y_train2)

# Initialize RandomForestClassifier with specified parameters
classifier = RandomForestClassifier(max_depth=None, n_estimators=200, random_state=42)

# Fit the classifier to the resampled data
classifier.fit(X_train_adasyn, y_train_adasyn)

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

# Define the RandomUnderSampler with specified parameters
undersample = RandomUnderSampler(sampling_strategy=0.75, random_state=42)

# Undersample the training data
X_train_undersample, y_train_undersample = undersample.fit_resample(X_train3, y_train3)

# Define the RandomForestClassifier with specified parameters
classifier = BalancedRandomForestClassifier(max_depth=30, n_estimators=300, random_state=42)

# Fit the classifier to the undersampled data
classifier.fit(X_train_undersample, y_train_undersample)

# Select the top features
selected_indices = indices[:optimal_num_features]
X_train_selected = X_train_after[:, selected_indices]
X_test_selected = X_test_after[:, selected_indices]

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Fit the model
    model.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Classification report
    class_report = classification_report(y_test, y_pred)

    # Print the evaluation metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Print confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    # Print classification report
    print("Classification Report:")
    print(class_report)

# Random Forest Model Evaluation with Selected Features
print("Random Forest Model without Feature Engineering or Balancing:")
evaluate_model(RandomForestClassifier(random_state=42), X_train_before, y_train_before, X_test_after, y_test_after)

print("\nRandom Forest Model with Feature Engineering but No Balancing:")
evaluate_model(RandomForestClassifier(random_state=42), X_train_selected, y_train_after, X_test_selected, y_test_after)

print("\nRandom Forest Model with SMOTE Resampling:")
evaluate_model(RandomForestClassifier(random_state=42), X_train_smote, y_train_smote, X_test_after, y_test_after)

print("\nRandom Forest Model with ADASYN Resampling:")
evaluate_model(RandomForestClassifier(random_state=42), X_train_adasyn, y_train_adasyn, X_test_after, y_test_after)

print("\nRandom Forest Model with Random Undersampling:")
evaluate_model(RandomForestClassifier(random_state=42), X_train_undersample, y_train_undersample, X_test_after, y_test_after)

# SVM Model Evaluation
print("SVM Model without Feature Engineering or Balancing:")
evaluate_model(SVC(random_state=42), X_train_before, y_train_before, X_test_after, y_test_after)

print("\nSVM Model with Feature Engineering but No Balancing:")
evaluate_model(SVC(random_state=42), X_train_selected, y_train_after, X_test_selected, y_test_after)

print("\nSVM Model with SMOTE Resampling:")
evaluate_model(SVC(random_state=42), X_train_smote, y_train_smote, X_test_after, y_test_after)

print("\nSVM Model with ADASYN Resampling:")
evaluate_model(SVC(random_state=42), X_train_adasyn, y_train_adasyn, X_test_after, y_test_after)

print("\nSVM Model with Random Undersampling:")
evaluate_model(SVC(random_state=42), X_train_undersample, y_train_undersample, X_test_after, y_test_after)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Define the MLP model
def create_mlp_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# MLP Model Evaluations
print("MLP Model without Feature Engineering or Balancing:")
mlp_model = create_mlp_model(input_shape=(X_train_before.shape[1],))
mlp_model.fit(X_train_before, y_train_before, epochs=10, batch_size=32, verbose=0)
mlp_pred = (mlp_model.predict(X_test_after) > 0.5).astype("int32")
print(classification_report(y_test_after, mlp_pred, digits=4))

# MLP Model with Feature Engineering but No Balancing
print("\nMLP Model with Feature Engineering but No Balancing:")
mlp_model_fe = create_mlp_model(input_shape=(X_train_selected.shape[1],))
mlp_model_fe.fit(X_train_selected, y_train_after, epochs=10, batch_size=32, verbose=0)
mlp_pred_fe = (mlp_model_fe.predict(X_test_selected) > 0.5).astype("int32")
print(classification_report(y_test_after, mlp_pred_fe, digits=4))


print("\nMLP Model with SMOTE Resampling:")
mlp_model_smote = create_mlp_model(input_shape=(X_train_smote.shape[1],))
mlp_model_smote.fit(X_train_smote, y_train_smote, epochs=10, batch_size=32, verbose=0)
mlp_pred_smote = (mlp_model_smote.predict(X_test_after) > 0.5).astype("int32")
print(classification_report(y_test_after, mlp_pred_smote, digits=4))

print("\nMLP Model with ADASYN Resampling:")
mlp_model_adasyn = create_mlp_model(input_shape=(X_train_adasyn.shape[1],))
mlp_model_adasyn.fit(X_train_adasyn, y_train_adasyn, epochs=10, batch_size=32, verbose=0)
mlp_pred_adasyn = (mlp_model_adasyn.predict(X_test_after) > 0.5).astype("int32")
print(classification_report(y_test_after, mlp_pred_adasyn, digits=4))

print("\nMLP Model with Random Undersampling:")
mlp_model_undersample = create_mlp_model(input_shape=(X_train_undersample.shape[1],))
mlp_model_undersample.fit(X_train_undersample, y_train_undersample, epochs=10, batch_size=32, verbose=0)
mlp_pred_undersample = (mlp_model_undersample.predict(X_test_after) > 0.5).astype("int32")
print(classification_report(y_test_after, mlp_pred_undersample, digits=4))
