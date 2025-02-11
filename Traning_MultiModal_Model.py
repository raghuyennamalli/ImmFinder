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

