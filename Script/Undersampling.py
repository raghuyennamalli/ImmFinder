# -*- coding: utf-8 -*-
"""Copy of undersampling.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QdBx7vrrAZMhUKJnI7J_deHjMoM_5By3
"""

import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

"""# NEAR MISS

"""

import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import NearMiss

# Load the dataset from CSV, specifying low_memory=False to handle mixed types
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/Final_wd/genomic_variations_wd.csv', low_memory=False)

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column (except 'gene_category' which is the target)
label_encoders = {}
for col in categorical_columns:
    if col != 'gene_category':  # Skip encoding for the target column
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

# Handle missing values (NaNs or infs)
data = data.dropna()

# Separate features (X) and target (y) where 'gene_category' is the target
X = data.drop(columns=['gene_category'])
y = data['gene_category']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply NearMiss for undersampling the majority class
nearmiss = NearMiss(version=2)  # NearMiss version 2
X_resampled, y_resampled = nearmiss.fit_resample(X_train, y_train)

# Convert the resampled data back to a DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['gene_category'] = y_resampled  # Add the target column 'gene_category' back

# Reverse the encoding for the categorical columns
for col, le in label_encoders.items():
    resampled_data[col] = le.inverse_transform(resampled_data[col])

# Save the modified dataset to a new CSV file
resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/Final_wd/genomic_variations_wd_nearmiss.csv', index=False)

# Train a RandomForest classifier on the resampled dataset
clf = RandomForestClassifier(random_state=42)
clf.fit(X_resampled, y_resampled)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

print("X shape:", X.shape)
print("y shape:", y.shape)

"""# TOMEKLINK"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import TomekLinks

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/cleaned_files/cleaned_processed_consolidated_INS_final.csv', low_memory=False)

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column (except 'category' which is the target)
label_encoders = {}
for col in categorical_columns:
    if col != 'gene_category':  # Skip encoding for the target column
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

# Handle missing values
data = data.dropna()

# Separate features (X) and target (y)
X = data.drop(columns=['gene_category'])
y = data['gene_category']

# Check the dataset size
print(f"Number of samples in the dataset: {X.shape[0]}")

# Ensure the dataset has enough samples for splitting
if X.shape[0] > 1:
    # Use stratified splitting to maintain the proportion of immune/non-immune in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    # Apply Tomek Links for undersampling the majority class
    tomek = TomekLinks()
    X_resampled, y_resampled = tomek.fit_resample(X_train, y_train)

    # Convert resampled data to a DataFrame
    resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_data['gene_category'] = y_resampled

    # Save the undersampled dataset
    resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/resample/ins_tomek.csv', index=False)

    # Train a RandomForest classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_resampled, y_resampled)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    print(classification_report(y_test, y_pred))
else:
    print("Not enough samples to split the dataset.")

"""# clustercentroid

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import ClusterCentroids

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/genomics/new_merged_var/gen_merged_with_variation.csv', low_memory=False)

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column (except 'category' which is the target)
label_encoders = {}
for col in categorical_columns:
    if col != 'gene_category':  # Skip encoding for the target column
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

# Handle missing values
data = data.dropna()

# Separate features (X) and target (y)
X = data.drop(columns=['gene_category'])
y = data['gene_category']

# Check the dataset size before splitting
print(f"Original number of samples: {X.shape[0]}")

# Ensure the dataset has enough samples for splitting
if X.shape[0] > 1:
    # Use stratified splitting to maintain the proportion of immune/non-immune in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    # Check the size of the training set
    print(f"Number of samples in training set before Cluster Centroids: {X_train.shape[0]}")

    # Apply Cluster Centroids for undersampling the majority class
    cluster_centroids = ClusterCentroids()
    X_resampled, y_resampled = cluster_centroids.fit_resample(X_train, y_train)

    # Check the size of the resampled dataset
    print(f"Number of samples in training set after Cluster Centroids: {X_resampled.shape[0]}")

    # Convert resampled data to a DataFrame
    resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_data['gene_category'] = y_resampled
# Reverse the encoding for the categorical columns
    for col, le in label_encoders.items():
      resampled_data[col] = le.inverse_transform(resampled_data[col])

    # Save the undersampled dataset
    resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/genomics/new_merged_var/gen_merged_with_variation_cc.csv', index=False)

    # Train a RandomForest classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_resampled, y_resampled)

    # Predict on the test set
    y_pred = clf.predict(X_test)
    # Evaluate the model
    print(classification_report(y_test, y_pred))
else:
    print("Not enough samples to split the dataset.")

"""# Neighbourhood cleaning"""

!pip install imbalanced-learn

!pip install --upgrade imbalanced-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/cleaned_files/cleaned_processed_consolidated_INS_final.csv', low_memory=False)

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column (except 'category' which is the target)
label_encoders = {}
for col in categorical_columns:
    if col != 'gene_category':  # Skip encoding for the target column
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

# Handle missing values by dropping any rows with NaN values
data = data.dropna()

# Separate features (X) and target (y)
X = data.drop(columns=['gene_category'])
y = data['gene_category']

# Check the dataset size before splitting
print(f"Original number of samples: {X.shape[0]}")

# Ensure the dataset has enough samples for splitting
if X.shape[0] > 1:
    # Use stratified splitting to maintain the proportion of immune/non-immune in train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    # Check the size of the training set
    print(f"Number of samples in training set before cleaning: {X_train.shape[0]}")

    # Reset index before applying nearest neighbors
    X_combined = X_train.reset_index(drop=True)
    y_combined = y_train.reset_index(drop=True)

    # Identify the minority and majority classes
    majority_class = y_combined.value_counts().idxmax()
    minority_class = y_combined.value_counts().idxmin()

    # Get indices of majority and minority samples
    majority_indices = y_combined[y_combined == majority_class].index
    minority_indices = y_combined[y_combined == minority_class].index

    # Initialize the resampled data with minority samples
    X_resampled = X_combined.loc[minority_indices].copy()
    y_resampled = y_combined.loc[minority_indices].copy()

    # Further remove noisy majority instances using k-NN
    # Fit nearest neighbors to the minority class samples
    neighbors = NearestNeighbors(n_neighbors=5)  # Use 5 neighbors
    neighbors.fit(X_combined)

    # Find neighbors of majority class samples
    distances, indices = neighbors.kneighbors(X_combined.loc[majority_indices])

    # Remove majority instances that are surrounded by too many minority instances
    for i, idx in enumerate(majority_indices):
        if np.sum(y_combined.loc[indices[i]] == minority_class) >= 3:  # Threshold: majority instances surrounded by at least 3 minority instances
            continue
        else:
            # Use pd.concat instead of append
            X_resampled = pd.concat([X_resampled, X_combined.loc[[idx]]], ignore_index=True)
            y_resampled = pd.concat([y_resampled, y_combined.loc[[idx]]], ignore_index=True)

    # Check the size of the resampled dataset
    print(f"Number of samples in training set after cleaning: {X_resampled.shape[0]}")

    # Save the resampled dataset
    resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_data['gene_category'] = y_resampled
    resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/resample/ins_resampled_cleaning.csv', index=False)

    # Train a RandomForest classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_resampled, y_resampled)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    print(classification_report(y_test, y_pred))
else:
    print("Not enough samples to split the dataset.")

"""#EditedNearestNeighbours"""

import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.preprocessing import LabelEncoder

# Load the dataset from CSV (forcing mixed-type columns to string to avoid DtypeWarning)
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/cleaned_files/cleaned_processed_consolidated_INS_final.csv', dtype=str)

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column (except 'category' which is the target)
label_encoders = {}
for col in categorical_columns:
    if col != 'gene_category':  # Skip encoding for the target column
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

# Separate features (X) and target (y)
X = data.drop('gene_category', axis=1)  # All columns except 'category' as features
y = data['gene_category']  # 'category' as target variable

# Initialize EditedNearestNeighbours for undersampling
enn = EditedNearestNeighbours()

# Apply undersampling on the dataset
X_resampled, y_resampled = enn.fit_resample(X, y)

# Combine resampled features and target into a new dataframe
resampled_data = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name='gene_category')], axis=1)

# Save the resampled dataset to a new CSV file
resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/resample/ins_new_ENN_balanced.csv', index=False)

print("Dataset has been undersampled and saved to 'ins_new_ENN_balanced.csv'.")

"""#OneSidedSelection"""

import pandas as pd
from imblearn.under_sampling import OneSidedSelection

# Load the dataset from CSV (replace 'input_file.csv' with your actual file path)
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/GSE159268_new.csv')
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column (except 'category' which is the target)
label_encoders = {}
for col in categorical_columns:
    if col != 'category':  # Skip encoding for the target column
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
# Separate features (X) and target (y)
X = data.drop('category', axis=1)  # All columns except 'category' are treated as features
y = data['category']  # 'category' is the target variable (contains 'immune' and 'non-immune')

# Initialize OneSidedSelection for undersampling
oss = OneSidedSelection()

# Apply undersampling on the dataset
X_resampled, y_resampled = oss.fit_resample(X, y)


# Combine resampled features and target into a new dataframe
resampled_data = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name='category')], axis=1)

# Save the resampled dataset to a new CSV file (replace 'output_file.csv' with your desired path)
resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/resample/GSE159268_new_OSS.csv', index=False)


print("Dataset has been undersampled using OneSidedSelection and saved.")

import pandas as pd
from imblearn.under_sampling import OneSidedSelection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/cleaned_files/cleaned_processed_consolidated_INS_final.csv', dtype='str')

# Identify and encode categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    if col != 'gene_category':  # Skip encoding for the target column
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

# Separate features and target, map category to 1 ('immune') and 0 ('non-immune')
X = data.drop(columns=['gene_category'])  # All columns except 'category' are features
y = data['gene_category'].map({'immune': 1, 'non-immune': 0})  # Encode target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize OneSidedSelection (OSS) without preserving the minority class
oss = OneSidedSelection(random_state=42, n_neighbors=1)

# Apply OSS on the training data
X_resampled, y_resampled = oss.fit_resample(X_train, y_train)

# Convert the resampled labels back to their original form (1: 'immune', 0: 'non-immune')
y_resampled = pd.Series(y_resampled).map({1: 'immune', 0: 'non-immune'})

# Convert the resampled data back to a DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['gene_category'] = y_resampled

# Check the new class distribution
print("Resampled class distribution:\n", resampled_data['gene_category'].value_counts())

# Save the resampled dataset to a new CSV file
resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/resample/ins_new_OSS_balanced.csv', index=False)

print("Dataset has been balanced using OneSidedSelection and saved.")

"""#RepeatedEditedNearestNeighbours"""

import pandas as pd
from imblearn.under_sampling import RepeatedEditedNearestNeighbours

# Load the dataset from CSV (replace 'input_file.csv' with your actual file path)
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/GSE159268_new.csv')
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column (except 'category' which is the target)
label_encoders = {}
for col in categorical_columns:
    if col != 'category':  # Skip encoding for the target column
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
# Separate features (X) and target (y)
X = data.drop('category', axis=1)  # All columns except 'category' are treated as features
y = data['category']  # 'category' is the target variable (contains 'immune' and 'non-immune')

# Initialize RepeatedEditedNearestNeighbours for undersampling
renn = RepeatedEditedNearestNeighbours()

# Apply undersampling on the dataset
X_resampled, y_resampled = renn.fit_resample(X, y)

# Combine resampled features and target into a new dataframe
resampled_data = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name='category')], axis=1)

# Save the resampled dataset to the same CSV file (overwrite)
resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/resample/GSE159268_new_REN.csv', index=False)

print("Dataset has been undersampled using RepeatedEditedNearestNeighbours and saved.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RepeatedEditedNearestNeighbours

# Load the dataset from CSV
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/GSE159268_new.csv', low_memory=False)

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column except 'category'
label_encoders = {}
for col in categorical_columns:
    if col != 'category':  # Skip encoding for the target column 'category'
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

# Handle missing values by dropping rows with missing values
data = data.dropna()

# Separate features (X) and target (y) where 'category' is the target
X = data.drop(columns=['category'])  # All columns except the target column 'category'
y = data['category']  # The target column remains 'immune' and 'non-immune'

# Check initial class distribution
print("Initial class distribution:\n", y.value_counts())

# Encode the target variable to numeric for resampling
y_encoded = y.map({'immune': 1, 'non-immune': 0})

# Split data into training and test sets with stratified sampling to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Initialize RepeatedEditedNearestNeighbours for undersampling
renn = RepeatedEditedNearestNeighbours()

# Apply RepeatedEditedNearestNeighbours for undersampling the majority class
X_resampled, y_resampled_encoded = renn.fit_resample(X_train, y_train)

# Convert the resampled data back to a DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)

# Decode the target variable back to original string labels
resampled_data['category'] = y_resampled_encoded.map({1: 'immune', 0: 'non-immune'})

# Check the class distribution after resampling
print("Resampled class distribution:\n", resampled_data['category'].value_counts())

# Save the resampled dataset to a new CSV file
resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/resample/GSE159268_new_REN_2.csv', index=False)

print("Resampled dataset saved successfully.")

import pandas as pd
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset from CSV (replace 'your_dataset.csv' with the correct file)
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/cleaned_files/cleaned_processed_consolidated_INS_final.csv', dtype='str')

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column (except 'category')
label_encoders = {}
for col in categorical_columns:
    if col != 'gene_category':  # Skip encoding for the target column
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

# Separate features (X) and target (y) where 'category' is the target
X = data.drop(columns=['gene_category'])  # All columns except the target column 'category'
y = data['gene_category']  # The target column 'category'

# Ensure the target is not encoded as 0/1, but as 'immune' and 'non-immune'
# Assuming immune is minority and non-immune is majority
y = y.map({'immune': 1, 'non-immune': 0})

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the RandomForestClassifier as the base estimator for RENN
clf_rf = RandomForestClassifier(random_state=42)

# Initialize RepeatedEditedNearestNeighbours for undersampling
renn = RepeatedEditedNearestNeighbours()

# Apply undersampling to balance the dataset
X_resampled, y_resampled = renn.fit_resample(X_train, y_train)

# Convert the resampled labels back to 'immune' and 'non-immune'
y_resampled = pd.Series(y_resampled).map({1: 'immune', 0: 'non-immune'})

# Convert the resampled data back to a DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['gene_category'] = y_resampled  # Add the target column 'category' back

# Check the new class distribution after RENN
print("Resampled class distribution:\n", resampled_data['gene_category'].value_counts())

# Save the balanced dataset to a new CSV file
resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/resample/ins_new_RENN_balanced.csv', index=False)

print("Dataset has been balanced using RENN and saved.")

"""# AllKNN"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import AllKNN

# Load the dataset from CSV
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/GSE159268_new.csv')

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column (except 'category' which is the target)
label_encoders = {}
for col in categorical_columns:
    if col != 'category':  # Skip encoding for the target column
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

# Separate features (X) and target (y)
X = data.drop('category', axis=1)  # All columns except 'category' are treated as features
y = data['category']  # 'category' is the target variable (contains 'immune' and 'non-immune')

# Initialize AllKNN for undersampling
allknn = AllKNN()

# Apply undersampling on the dataset
X_resampled, y_resampled = allknn.fit_resample(X, y)

# Combine resampled features and target into a new dataframe
resampled_data = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name='category')], axis=1)

# Save the resampled dataset to the same CSV file (overwrite)
resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/resample/GSE159268_new_AIIKNN.csv', index=False)

print("Dataset has been undersampled using AllKNN and saved.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import AllKNN

# Load the dataset from CSV
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/GSE159268_new.csv', low_memory=False)

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column except 'category'
label_encoders = {}
for col in categorical_columns:
    if col != 'category':  # Skip encoding for the target column 'category'
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

# Handle missing values by dropping rows with missing values
data = data.dropna()

# Separate features (X) and target (y) where 'category' is the target
X = data.drop(columns=['category'])  # All columns except the target column 'category'
y = data['category']  # The target column remains 'immune' and 'non-immune'

# Encode the target variable to numeric for resampling
y_encoded = y.map({'immune': 1, 'non-immune': 0})

# Check initial class distribution
print("Initial class distribution:\n", y.value_counts())

# Split data into training and test sets with stratified sampling to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Initialize AllKNN for undersampling
allknn = AllKNN()

# Apply AllKNN for undersampling the majority class
X_resampled, y_resampled_encoded = allknn.fit_resample(X_train, y_train)

# Convert the resampled data back to a DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)

# Decode the target variable back to original string labels
resampled_data['category'] = y_resampled_encoded.map({1: 'immune', 0: 'non-immune'})

# Check the class distribution after resampling
print("Resampled class distribution:\n", resampled_data['category'].value_counts())

# Save the resampled dataset to a new CSV file
resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/resample/GSE159268_new_AIIKNN_2.csv', index=False)

# Train a RandomForest classifier on the resampled dataset
clf_resampled = RandomForestClassifier(random_state=42)
clf_resampled.fit(X_resampled, y_resampled_encoded)

# Predict on the test set
y_pred_encoded = clf_resampled.predict(X_test)

# Decode predictions back to original string labels
y_pred = pd.Series(y_pred_encoded).map({1: 'immune', 0: 'non-immune'})

print(classification_report(y_test.map({1: 'immune', 0: 'non-immune'}), y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import AllKNN

# Load the dataset from CSV
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/cleaned_files/cleaned_processed_consolidated_INS_final.csv', low_memory=False)


categorical_columns = data.select_dtypes(include=['object']).columns

# Apply LabelEncoder
label_encoders = {}
for col in categorical_columns:
    if col != 'gene_category':
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])


# Separate features
X = data.drop(columns=['gene_category'])
y = data['gene_category']


print("Initial class distribution:\n", y.value_counts())

# Encode
y_encoded = y.map({'immune': 1, 'non-immune': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)


allknn = AllKNN()

# AllKNN
X_resampled, y_resampled_encoded = allknn.fit_resample(X_train, y_train)
# Convert
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
# Decode
resampled_data['gene_category'] = y_resampled_encoded.map({1: 'immune', 0: 'non-immune'})
print("Resampled class distribution:\n", resampled_data['gene_category'].value_counts())

# Save
resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/resample/ins_new_AllKNN_resampled.csv', index=False)

print("Resampled dataset saved successfully.")

"""# InstanceHardnessThreshold"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import InstanceHardnessThreshold

# Load the dataset from CSV
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/GSE159268_new.csv')

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column (except 'category' which is the target)
label_encoders = {}
for col in categorical_columns:
    if col != 'category':  # Skip encoding for the target column
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

# Separate features (X) and target (y)
X = data.drop('category', axis=1)  # All columns except 'category' are treated as features
y = data['category']  # 'category' is the target variable (contains 'immune' and 'non-immune')

# Encode the target labels (immune/non-immune) into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Initialize InstanceHardnessThreshold for undersampling
iht = InstanceHardnessThreshold()

# Apply undersampling on the dataset
X_resampled, y_resampled = iht.fit_resample(X, y_encoded)

# Convert the resampled target labels back to their original string format
y_resampled = label_encoder.inverse_transform(y_resampled)

# Combine resampled features and target into a new dataframe
resampled_data = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name='category')], axis=1)

# Save the resampled dataset to the same CSV file (overwrite)
resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/resample/GSE159268_new_IHT.csv', index=False)

print("Dataset has been undersampled using Instance Hardness Threshold and saved.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import InstanceHardnessThreshold

# Load the dataset from CSV
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/GSE159268_new.csv', low_memory=False)

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column (including 'category' which is the target)
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Handle missing values by dropping rows with missing values
data = data.dropna()

# Separate features (X) and target (y) where 'category' is the target
X = data.drop(columns=['category'])  # All columns except the target column 'category'
y = data['category']  # The target column 'category' (immune: 1, non-immune: 0)

# Ensure y is integer-encoded
y = y.astype(int)

# Split data into training and test sets with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize the RandomForestClassifier to be used in InstanceHardnessThreshold
clf_rf = RandomForestClassifier(random_state=42)

# Apply InstanceHardnessThreshold (IHT) for undersampling
iht = InstanceHardnessThreshold(estimator=clf_rf, random_state=42)

# Fit and resample the training set to reduce the majority class
X_resampled, y_resampled = iht.fit_resample(X_train, y_train)

# Convert the resampled data back to a DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['category'] = y_resampled  # Add the target column 'category' back

# Save the modified dataset to a new CSV file
resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/resample/GSE159268_new_IHT_2.csv', index=False)

# Train a RandomForest classifier on the resampled dataset
clf_resampled = RandomForestClassifier(random_state=42)
clf_resampled.fit(X_resampled, y_resampled)

# Predict on the test set
y_pred = clf_resampled.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import InstanceHardnessThreshold

# Load the dataset from CSV
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/cleaned_files/cleaned_processed_consolidated_INS_final.csv', low_memory=False)

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply LabelEncoder to each categorical column except 'category'
label_encoders = {}
for col in categorical_columns:
    if col != 'gene_category':  # Skip encoding for the target column 'category'
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

# Handle missing values by dropping rows with missing values
data = data.dropna()

# Separate features (X) and target (y) where 'category' is the target
X = data.drop(columns=['gene_category'])  # All columns except the target column 'category'
y = data['gene_category']  # The target column remains 'immune' and 'non-immune'

# Encode the target variable to numeric for resampling
y_encoded = y.map({'immune': 1, 'non-immune': 0})

# Check initial class distribution
print("Initial class distribution:\n", y.value_counts())

# Split data into training and test sets with stratified sampling to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Initialize the RandomForestClassifier to be used in InstanceHardnessThreshold
clf_rf = RandomForestClassifier(random_state=42)

# Fit the classifier on the training set to ensure it has learned the data distribution
clf_rf.fit(X_train, y_train)

# Apply InstanceHardnessThreshold (IHT) for undersampling the majority class
iht = InstanceHardnessThreshold(estimator=clf_rf, random_state=42)

# Fit and resample the training set to reduce the majority class (non-immune)
X_resampled, y_resampled_encoded = iht.fit_resample(X_train, y_train)

# Convert the resampled data back to a DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)

# Decode the target variable back to original string labels
resampled_data['gene_category'] = y_resampled_encoded.map({1: 'immune', 0: 'non-immune'})

# Check the class distribution after resampling
print("Resampled class distribution:\n", resampled_data['gene_category'].value_counts())

# Save the resampled dataset to a new CSV file
resampled_data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/resample/ins_new_IHT_2.csv', index=False)

# Train a RandomForest classifier on the resampled dataset
clf_resampled = RandomForestClassifier(random_state=42)
clf_resampled.fit(X_resampled, y_resampled_encoded)

# Predict on the test set
y_pred_encoded = clf_resampled.predict(X_test)

# Decode predictions back to original string labels
y_pred = pd.Series(y_pred_encoded).map({1: 'immune', 0: 'non-immune'})

# Evaluate the model
print(classification_report(y_test.map({1: 'immune', 0: 'non-immune'}), y_pred))