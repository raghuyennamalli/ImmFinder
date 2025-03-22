ImmFinder: A Multimodal Fully Connected Neural Network for Immune Gene Classification in Cattle
Authors
Menaka Thambiraja1, Pavinap Priyaa Karthikeyan1$, Mezya Sezen1$, Shricharan Senthilkumar1$, Dheer Singh2, Suneel Kumar Onteru2*, Ragothaman M. Yennamalli1*
Corresponding authors: ragothaman@scbt.sastra.edu
Table of Contents
•	Description
•	Contents
•	Methods
•	Results
•	Installation
•	Usage
•	Model Usage
•	File Information
•	Contact
Description
ImmFinder is a multimodal fully connected neural network (FCNN) framework developed to classify immune and non-immune genes in cattle by integrating genomic and transcriptomic datasets. Traditional methods for immune gene classification rely on manual curation and conventional bioinformatics tools, which can be time-consuming and labor-intensive. ImmFinder leverages deep learning to automate and enhance this classification process.
Contents
•	Model: The trained FCNN model for immune gene classification.
•	Datasets: The datasets include raw genomic and transcriptomic data, along with preprocessed datasets split into training (60%), validation (20%), and testing (20%) sets.
•	Result: Performance metrics including accuracy, F1-score, precision, recall, and AUC-ROC scores for validated datasets.
•	Script: Scripts for data preprocessing, model training, and validation.

Methods
•	Data Sources: Structural variant data between the whole genome comparative study of Bos taurus and Bos indicus. Bovine Gene expression data for infected and uninfected sets are sourced from the Gene Expression Omnibus (GEO).
•	Data Preprocessing: Balancing the class (True positive and True negative), Encoding and standardization of the datasets. Splitting into training (60%), validation (20%), and testing (20%) sets.
•	Model Architecture: Two-branch FCNN with 12 fully connected dense layers (ReLU activation, 10% dropout). Concatenation of branches followed by a dense layer with two neurons. Final sigmoid output layer with one neuron (total of 27 neurons).
•	Optimization Techniques: L2 regularization, focal loss, Adam optimizer. Trained for 50 epochs with a batch size of 32.

Results
•	Accuracy: 85.67%
•	F1-score: 0.85
•	Precision: 0.86
•	Recall: 0.85
•	AUC-ROC: 0.9250 (test set), 0.9264 (validation set)

These results highlight the robustness of ImmFinder in immune gene classification and its potential to advance functional genomics in cattle.

Installation
To use ImmFinder, follow these steps:
1.	Clone the repository:
git clone https://github.com/raghuyennamalli/ImmFinder.git
cd ImmFinder
2. Install dependencies:
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn

Usage
1. Handling the class imbalance in the dataset:
python Undersampling.py
2. Train and Evaluate the model:
python Multi_Modal_Model.py

Model Usage
The deposited ImmFinder trained model (.keras format) can be utilized in multiple ways:
•	Direct Inference Without Training: You can load the trained model and classify immune vs. non-immune genes without retraining, saving time and computational resources.
•	Reproducibility & Benchmarking: You can verify the model's performance and compare it with other classification models.
•	Fine-Tuning & Transfer Learning: You can fine-tune the model with their own datasets for improved classification accuracy in specific cattle populations or other species.
•	Understanding Model Architecture & Performance: You can analyze the model's layer configurations, weight distributions, and intermediate activations to gain insights into the prediction process.

To Load the Trained Model
from tensorflow.keras.models import load_model
model = load_model("Best_validated_multimodal_model.keras")
model.summary()


File Information
•	Format: Keras format for trained and validated models. CSV files for raw, splitted datasets and validated results. Python for data preprocessing, and model development.
•	Data Source: The structural variant data were identified by performing comparative genomic study between Bos taurus and Bos indicus using GSAlign and SyRI tool. The whole Genomic data of the both the cattle downloaded from NCBI. The bovine Gene expression data for infected and uninfected cases were extracted from GEO.
•	Implementation: ImmFinder is implemented in Python.

Contact
For any queries or further information, please contact:
Ragothaman M. Yennamalli
Senior Assistant Professor, Department of Bioinformatics, School of Chemical and Biotechnology, SASTRA Deemed University
Email: ragothaman@scbt.sastra.edu
ORCID: 0000-0002-3327-1582


