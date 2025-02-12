from google.colab import drive
drive.mount('/content/drive')

"""#INNATE IMMUNE, SPECIES, INFECTION COLUMN"""

import pandas as pd
import shutil
import os

# Read
df = pd.read_excel("/content/drive/MyDrive/ImmFinder/Transcriptome/GSE141962/Table.xlsx")
df_innate = pd.read_excel("/content/drive/MyDrive/ImmFinder/innate_immgen_annnot.xlsx")
table = pd.DataFrame(df)

#set creation of gene symbol
innate_genes = set(df_innate['Gene Symbol'])

# innate_immune column
table["innate_immune"] = table["Gene Symbol"].apply(
    lambda gene: 'unannotated' if pd.isna(gene) or gene in [0, '#N/A']
    else (True if gene in innate_genes else False))

# species column
species_name = "Bos taurus"
table["species"] = species_name

# condition column
condition_value = "MAP Infection"
table["condition"] = condition_value

# Save to CSV
table.to_csv("GSE24048_edited.csv", index=False)
shutil.copy("GSE24048_edited.csv", "/content/drive/MyDrive/ImmFinder/Transcriptome/GSE141962_edited.csv")

"""#ADAPTIVE, category COLUMN"""

import pandas as pd
import shutil
import os

# Read data
df = pd.read_csv("/content/drive/MyDrive/ImmFinder/transcriptomics/GSE152959_new.csv")
df_adaptive = pd.read_csv("/content/drive/MyDrive/ImmFinder/Raw_HF_data.csv")

# Convert to DataFrame
table = pd.DataFrame(df)

# Create a set of adaptive immune genes
adaptive_genes = set(df_adaptive['Symbol'])

# Create a dictionary mapping gene symbols to their 'keyword' value in df_adaptive
adaptive_dict = df_adaptive.set_index('Symbol')['keyword_found'].to_dict()

# Ensure all entries in 'Gene Symbol' column are strings, and then apply the adaptive column creation
table['adaptive'] = table['Gene Symbol'].apply(lambda x: adaptive_dict.get(str(x).strip(), 'Not Present') if pd.notna(x) and str(x).strip() != '' else 'Not Present')

# Define the categorize function to check for 'TRUE' or 'FALSE', ignoring case and whitespace
def categorize(row):
    innate_immune = str(row['innate_immune']).strip().lower()
    adaptive = str(row['adaptive']).strip().lower()

    if innate_immune == 'true' or adaptive == 'true':
        return 'immune'
    elif innate_immune == 'false' or adaptive == 'false':
        return 'non-immune'
    else:
        return 'NA'

# Apply the categorization function
table['category'] = table.apply(categorize, axis=1)

# Save the edited DataFrame to a CSV file
table.to_csv("GSE107366_new.csv", index=False)

# Copy the CSV to the desired location
shutil.copy("GSE107366_new.csv", "/content/drive/MyDrive/ImmFinder/cleaned-transcriptome/GSE107366_new.csv")

"""#True Negative (TN), True Positive (TP) COUNTS"""

import pandas as pd

# Load the CSV file into a DataFrame
file_path ='/content/drive/MyDrive/ImmFinder/Final_wd/genomic_variations_wd_cc.csv'# Replace with your file path
df = pd.read_csv(file_path)

# Count occurrences in the 'category' column
counts = df['gene_category'].value_counts(dropna=False)  # dropna=False includes NaN values in the count

# Get counts for each specific value
non_immune_count = counts.get('non-immune', 0)
immune_count = counts.get('immune', 0)
na_count = counts.get('NA', 0) + counts.get(pd.NA, 0) + counts.get(None, 0) + df['gene_category'].isna().sum()
total_count = df['gene_category'].shape[0]
# Print the results
print(f"Non-immune count: {non_immune_count}")
print(f"Immune count: {immune_count}")
print(f"NA count: {na_count}")
print(total_count)

"""#CLEANED FILES - removing NAs"""

import pandas as pd

# Load the CSV file into a DataFrame
file_path = '/content/drive/MyDrive/ImmFinder/transcriptomics/processed/GSE152959_new_padj.csv'                  # Replace with your file path
df = pd.read_csv(file_path)

# Remove rows where 'category' is 'NA' or missing
cleaned_df = df[~df['category'].isin(['NA']) & df['category'].notna()]

# Save the cleaned DataFrame to a new CSV file
output_file_path = '/content/drive/MyDrive/ImmFinder/transcriptomics/cleaned/GSE152959_new_padj_cleaned.csv'  # Specify the output file path
cleaned_df.to_csv(output_file_path, index=False)

print(f"Cleaned data saved to {output_file_path}")

"""#2fold up and downregulated genes extract"""

import pandas as pd

# Load the dataset
file_path = '/content/drive/MyDrive/ImmFinder/transcriptomics/cleaned/GSE152959_new_padj_cleaned.csv'
df = pd.read_csv(file_path)

# Check if 'log2FoldChange' column exists in the DataFrame
if 'log2FoldChange' in df.columns:
    # Convert 'log2FoldChange' column to numeric, set errors='coerce' to handle non-numeric values
    df['log2FoldChange'] = pd.to_numeric(df['log2FoldChange'], errors='coerce')

    # Check for duplicates
    duplicates = df.duplicated()

    if duplicates.any():
        print(f"Found {duplicates.sum()} duplicate rows in the dataset.")

        # Optionally remove duplicates
        df_no_duplicates = df.drop_duplicates()
        print("Duplicates removed.")
    else:
        df_no_duplicates = df
        print("No duplicates found in the dataset.")

    # Filter for 2-fold upregulated (log2FoldChange ≥ 2) or 2-fold downregulated (log2FoldChange ≤ -2)
    filtered_df = df_no_duplicates[(df_no_duplicates['log2FoldChange'] >= 2) | (df_no_duplicates['log2FoldChange'] <= -2)]

    # Save the filtered dataset to a new CSV file
    output_file_path = '/content/drive/MyDrive/ImmFinder/transcriptomics/cleaned/GSE152959_new_padj_cleaned_2fold.csv'           # Replace with desired output file path
    filtered_df.to_csv(output_file_path, index=False)

    print(f"Filtered dataset saved to {output_file_path}")
else:
    print("The 'log2FoldChange' column does not exist in the dataset. Please check the column names.")

#CODE FOR CHECKING DUPLICATE ROWS IN A DATASET
import pandas as pd

# Load the dataset
file_path = '/content/drive/MyDrive/ImmFinder/transcriptomics/processed/GSE62048_2hpi_new.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# Check for duplicates
duplicates = df.duplicated()

if duplicates.any():
    print(f"Found {duplicates.sum()} duplicate rows in the dataset.")


    print("Duplicates removed.")
else:
    print("No duplicates found in the dataset.")

"""#CALCULATE ADJUSTED PVALUE"""

import pandas as pd
from statsmodels.stats.multitest import multipletests

# Load the CSV file (replace 'input_file.csv' with your actual file path)
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/GSE152959_new.csv')

# Assuming your CSV has 'log2fc' and 'pvalue' columns
# If the column names are different, adjust accordingly
log2fc = data['log2FoldChange']
pvalues = data['Pvalue']

# Apply Benjamini-Hochberg (FDR) correction
data['Padjusted'] = multipletests(pvalues, method='fdr_bh')[1]

# Save the updated dataframe to a new CSV file (replace 'output_file.csv' with your desired path)
data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/GSE152959_new_padj.csv', index=False)

print("Adjusted p-values have been saved to 'output_file.csv'")

import pandas as pd

# Load the CSV file (replace 'input_file.csv' with your actual file path)
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/GSE152959_new_padj.csv')

# Replace empty strings or spaces in the 'category' column with 'NA'
data['category'] = data['category'].replace(r'^\s*$', 'NA', regex=True)
data['category'] = data['category'].fillna('NA')

# Overwrite the same CSV file with updated data
data.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/processed/GSE152959_new_padj.csv', index=False)

print("Empty spaces in the 'category' column have been replaced with 'NA' in the same file.")

"""MERGE FILES!!"""

import pandas as pd

# Reading the 7 CSV files
df1 = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/new_merged/GSE107366_new_2fold.csv')
df2 = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/new_merged/GSE141962_new_2fold.csv')
df3 = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/new_merged/GSE159268_new_2fold.csv')
df4 = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/new_merged/GSE167574_new_2fold.csv')
df5 = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/new_merged/GSE152959_new_padj_cleaned_2fold.csv')
df6 = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/new_merged/GSE62048_2hpi__2fold_cleaned.csv')
df7 = pd.read_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/new_merged/GSE62048_6hpi__2fold_cleaned.csv')

# Merging the datasets
merged_df = pd.concat([df1, df2, df3, df4, df5, df6, df7], ignore_index=True)

# Saving the merged dataset to a new CSV file
merged_df.to_csv('/content/drive/MyDrive/ImmFinder/transcriptomics/new_merged/wo_resample_merged_transcript.csv', index=False)

# Displaying the merged dataset
print(merged_df)

import pandas as pd
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/cleaned_files/cleaned_processed_consolidated_INS_final.csv')
data['variation'] = 'ins'

data.to_csv('/content/drive/MyDrive/ImmFinder/genomics/new_merged_var/gen_ins_with_variation.csv', index=False)
print(data.head())

import pandas as pd
data = pd.read_csv('/content/drive/MyDrive/ImmFinder/cleaned_files/cleaned_processed_consolidated_DEL_final.csv')
data['variation'] = 'del'

data.to_csv('/content/drive/MyDrive/ImmFinder/genomics/new_merged_var/gen_del_with_variation.csv', index=False)
print(data.head())

df1 = pd.read_csv('/content/drive/MyDrive/ImmFinder/genomics/new_merged_var/gen_ins_with_variation.csv')
df2 = pd.read_csv('/content/drive/MyDrive/ImmFinder/genomics/new_merged_var/gen_del_with_variation.csv')

# Merging the datasets
merged_df = pd.concat([df1, df2], ignore_index=True)

# Saving the merged dataset to a new CSV file
merged_df.to_csv('/content/drive/MyDrive/ImmFinder/genomics/new_merged_var/gen_merged_with_variation.csv', index=False)

# Displaying the merged dataset
print(merged_df)