
from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

genomic_data_columns = ['Accession', 'Chromosome', 'Orientation','Gene Type','Symbol', 'Protein length','qtlgene_type', 'QTL Class', 'HFgene_type', 'variation']

transcriptomic_data_columns = ['log2FoldChange', 'Pvalue', 'Padjusted', 'species', 'condition']

genomic_data=pd.read_csv("/content/drive/MyDrive/ImmFinder/Final_wd/genomic_variations_wd_cc.csv")
transcriptome=pd.read_csv("/content/drive/MyDrive/ImmFinder/final_merged_files/transcript_cc_resample_after_merged_gene_category.csv")

genomic_data_filtered = genomic_data[genomic_data_columns]
transcriptomic_data_filtered = transcriptome[transcriptomic_data_columns]

y_genomic = genomic_data['gene_category']
y_transcriptomic = transcriptome['gene_category']

genomic_data.shape[0]

transcriptome.shape[0]

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

# Step 1: Encode and Scale Data
ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

# Encode genomic dataset
genomic_data_encoded = ordinal_encoder.fit_transform(genomic_data_filtered)
genomic_data_encoded = np.nan_to_num(genomic_data_encoded, nan=0.0, posinf=0.0, neginf=0.0)

# Encode transcriptomic dataset
transcriptomic_data_encoded = ordinal_encoder.fit_transform(transcriptomic_data_filtered)
transcriptomic_data_encoded = np.nan_to_num(transcriptomic_data_encoded, nan=0.0, posinf=0.0, neginf=0.0)

# Scale genomic and transcriptomic datasets
scaler_genomic = StandardScaler()
scaler_transcriptomic = StandardScaler()

X_genomic_scaled = scaler_genomic.fit_transform(genomic_data_encoded)
X_transcriptomic_scaled = scaler_transcriptomic.fit_transform(transcriptomic_data_encoded)

# Step 2: Determine Common Size (Genomic Dataset Size)
common_size = len(genomic_data_filtered)

X_transcriptomic_ov, y_transcriptomic_ov = resample(
    X_transcriptomic_scaled,
    y_transcriptomic,
    replace=True,
    n_samples=common_size,
    random_state=42
)

# Verify new sizes
print("Genomic data size:", len(genomic_data_filtered))
print("Oversampled transcriptomic data size:", len(X_transcriptomic_ov))

# Split genomic data
X_genomic_train, X_genomic_temp, y_genomic_train, y_genomic_temp = train_test_split(
    X_genomic_scaled, y_genomic, test_size=0.4, random_state=42
)
X_genomic_val, X_genomic_test, y_genomic_val, y_genomic_test = train_test_split(
    X_genomic_temp, y_genomic_temp, test_size=0.5, random_state=42
)

X_transcriptomic_train, X_transcriptomic_temp, y_transcriptomic_train, y_transcriptomic_temp = train_test_split(
    X_transcriptomic_ov, y_transcriptomic_ov, test_size=0.4, random_state=42
)
X_transcriptomic_val, X_transcriptomic_test, y_transcriptomic_val, y_transcriptomic_test = train_test_split(
    X_transcriptomic_temp, y_transcriptomic_temp, test_size=0.5, random_state=42
)

# Verify Sizes
print(f"Genomic Train Size: {len(X_genomic_train)}, Transcriptomic Train Size: {len(X_transcriptomic_train)}")
print(f"Genomic Validation Size: {len(X_genomic_val)}, Transcriptomic Validation Size: {len(X_transcriptomic_val)}")
print(f"Genomic Test Size: {len(X_genomic_test)}, Transcriptomic Test Size: {len(X_transcriptomic_test)}")

# Step 5: Encode Labels
label_encoder = LabelEncoder()

# Encode genomic labels
y_genomic_train_encoded = label_encoder.fit_transform(y_genomic_train)
y_genomic_val_encoded = label_encoder.transform(y_genomic_val)
y_genomic_test_encoded = label_encoder.transform(y_genomic_test)

# Encode transcriptomic labels
y_transcriptomic_train_encoded = label_encoder.fit_transform(y_transcriptomic_train)
y_transcriptomic_val_encoded = label_encoder.transform(y_transcriptomic_val)
y_transcriptomic_test_encoded = label_encoder.transform(y_transcriptomic_test)

# Step 6: Define the Multi-Modal Model
input_genomic = Input(shape=(X_genomic_train.shape[1],), name="genomic_input")
genomic_branch = Dense(8, activation='relu', kernel_regularizer=l2(0.01))(input_genomic)
genomic_branch = Dropout(0.1)(genomic_branch)
genomic_branch = Dense(4, activation='relu', kernel_regularizer=l2(0.01))(genomic_branch)

input_transcriptomic = Input(shape=(X_transcriptomic_train.shape[1],), name="transcriptomic_input")
transcriptomic_branch = Dense(8, activation='relu', kernel_regularizer=l2(0.01))(input_transcriptomic)
transcriptomic_branch = Dropout(0.1)(transcriptomic_branch)
transcriptomic_branch = Dense(4, activation='relu', kernel_regularizer=l2(0.01))(transcriptomic_branch)

merged = Concatenate()([genomic_branch, transcriptomic_branch])
final_dense = Dense(2, activation='relu', kernel_regularizer=l2(0.01))(merged)
final_output = Dense(1, activation='sigmoid')(final_dense)

multi_modal_model = Model(inputs=[input_genomic, input_transcriptomic], outputs=final_output)
import tensorflow as tf

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    y_true = tf.cast(y_true, dtype=tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_exp = tf.exp(-bce)
    focal_loss = alpha * (1 - bce_exp) ** gamma * bce
    return tf.reduce_mean(focal_loss)

multi_modal_model.compile(optimizer="adam", loss=focal_loss, metrics=["accuracy"])

# Step 7: Train the Model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

from sklearn.utils.class_weight import compute_class_weight

# Step 1: Compute Class Weights for Genomic Labels
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_genomic_train_encoded),
    y=y_genomic_train_encoded
)

# Convert class weights to a dictionary format required by Keras
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print("Class Weights:", class_weight_dict)

# Step 2: Modify the `model.fit` call to include class weights
history = multi_modal_model.fit(
    {"genomic_input": X_genomic_train, "transcriptomic_input": X_transcriptomic_train},
    y=y_genomic_train_encoded.astype(np.float32),
    validation_data=(
        {"genomic_input": X_genomic_val, "transcriptomic_input": X_transcriptomic_val},
        y_genomic_val_encoded.astype(np.float32),
    ),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    class_weight=class_weight_dict  # Include class weights here
)

# Evaluate on training data
train_loss, train_accuracy = multi_modal_model.evaluate(
    [X_genomic_train, X_transcriptomic_train],
    y_genomic_train_encoded.astype(np.float32),
    verbose=0
)
print(f"Train Loss: {train_loss:.4f}")
print(f"Train Accuracy: {train_accuracy:.4f}")

# Evaluate on validation data
val_loss, val_accuracy = multi_modal_model.evaluate(
    [X_genomic_val, X_transcriptomic_val],
    y_genomic_val_encoded.astype(np.float32),
    verbose=0
)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Evaluate on test data
test_loss, test_accuracy = multi_modal_model.evaluate(
    [X_genomic_test, X_transcriptomic_test],
    y_genomic_test_encoded.astype(np.float32),
    verbose=0
)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

import matplotlib.pyplot as plt

# Extract accuracy and loss from the training history
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot Accuracy
plt.figure(figsize=(10, 6))
plt.plot(train_accuracy, label='Training Accuracy', color='blue')
plt.plot(val_accuracy, label='Validation Accuracy', color='green')
plt.title('Model Accuracy Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.show()

# Plot Loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='red')
plt.title('Model Loss Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True)
plt.show()

y_pred_prob = multi_modal_model.predict(
    {"genomic_input": X_genomic_test, "transcriptomic_input": X_transcriptomic_test}
)
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_genomic_test_encoded, y_pred_prob)

# Calculate F1 scores for all thresholds
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

# Get the optimal threshold (maximizing F1 score)
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal Threshold: {optimal_threshold:.4f}")

# Convert probabilities to binary predictions based on the optimal threshold
y_pred_optimized = (y_pred_prob >= optimal_threshold).astype(int)

from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Predict probabilities on the test dataset
y_pred_prob = multi_modal_model.predict(
    {"genomic_input": X_genomic_test, "transcriptomic_input": X_transcriptomic_test}
).ravel()

# Convert probabilities to binary predictions using a default threshold of 0.5
y_pred = (y_pred_prob >= 0.5).astype(int)

# AUC and ROC Curve
fpr, tpr, _ = roc_curve(y_genomic_test_encoded, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="blue")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Additional Metrics
f1 = f1_score(y_genomic_test_encoded, y_pred)
precision = precision_score(y_genomic_test_encoded, y_pred)
recall = recall_score(y_genomic_test_encoded, y_pred)
conf_matrix = confusion_matrix(y_genomic_test_encoded, y_pred)

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix graphically
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Immune", "Immune"], yticklabels=["Non-Immune", "Immune"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Optional: Precision-Recall Curve
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_genomic_test_encoded, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, marker=".", label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()

# Identify optimal threshold based on F1 Score
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)  # Avoid division by zero
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal Threshold: {optimal_threshold:.4f}")

from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# AUC and ROC Curve
fpr, tpr, _ = roc_curve(y_genomic_test_encoded, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="blue")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Additional Metrics
f1 = f1_score(y_genomic_test_encoded, y_pred)
precision = precision_score(y_genomic_test_encoded, y_pred)
recall = recall_score(y_genomic_test_encoded, y_pred)
conf_matrix = confusion_matrix(y_genomic_test_encoded, y_pred)

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate predictions
y_pred = multi_modal_model.predict(
    {"genomic_input": X_genomic_test, "transcriptomic_input": X_transcriptomic_test}
)
y_pred_classes = (y_pred > 0.5).astype(int)  # Convert probabilities to binary classes

# Generate the confusion matrix
cm = confusion_matrix(y_genomic_test_encoded, y_pred_classes)

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-Immune", "Immune"],
    yticklabels=["Non-Immune", "Immune"],
)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix", fontsize=16)
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming `y_genomic_test_encoded` is the true labels and `y_pred_prob` is the predicted probabilities

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_genomic_test_encoded, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="blue", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=1, label='Random Guess')
plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)
plt.show()

# Print AUC value
print(f"Area Under the Curve (AUC): {roc_auc:.4f}")

from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Get predictions for the validation set
y_val_pred_prob = multi_modal_model.predict(
    {"genomic_input": X_genomic_val, "transcriptomic_input": X_transcriptomic_val}
).flatten()  # Ensure the predictions are flattened for compatibility

# Convert probabilities to binary predictions using a threshold of 0.5
y_val_pred = (y_val_pred_prob >= 0.5).astype(int)

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_genomic_val_encoded, y_val_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color="blue")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Validation Set)")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Calculate additional metrics
f1 = f1_score(y_genomic_val_encoded, y_val_pred)
precision = precision_score(y_genomic_val_encoded, y_val_pred)
recall = recall_score(y_genomic_val_encoded, y_val_pred)
conf_matrix = confusion_matrix(y_genomic_val_encoded, y_val_pred)

# Print metrics
print(f"AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, cmap="Blues", interpolation="nearest")
plt.title("Confusion Matrix (Validation Set)")
plt.colorbar()
plt.xticks([0, 1], ["Non-Immune", "Immune"], fontsize=10)
plt.yticks([0, 1], ["Non-Immune", "Immune"], fontsize=10)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black", fontsize=12)
plt.show()

