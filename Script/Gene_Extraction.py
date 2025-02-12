
import pandas as pd

# Remove index_col if there's no extra column or index column at the starting
df_common=pd.read_csv("common_wd1.csv", index_col="Unnamed: 0")
df_protein_file = pd.read_csv("ind_chr1_prot.csv")


df_protein_file.shape

def map_keywords(str_sentence):
    list_keywords = [
        "immunoglobulin","immunoreceptor","autoimmune","TLR","IgG",
        "autoimmune","autophagy","immunogen","immune","innate","T-cell","NF-kappa", "antigen",
        "B-cell","lymphocyte","histocompatibility","CD24","CD4","LY96", "BCR",
        "IFIT3","PGLYRP1","NKG2D","UL16","leukocyte","cytokine", "interleukin","interferon"
        "antimicrobial peptide","beta-defensin 2","IL16","IL2","chemokine", "antibody"
    ]
    bool_found = bool(
        [
            i for i in list_keywords if i.lower() in str_sentence.lower()
        ]
    )
    return bool_found

df_protein_file["keyword_found"] = df_protein_file.apply(lambda row: map_keywords(row["Protein Name"]),axis=1)

df_protein_file[df_protein_file["keyword_found"]==True].shape

df_protein_file[df_protein_file["keyword_found"]==True].drop_duplicates().shape

df_protein_file["start_of_next_gene"] = df_protein_file["Start"].shift(-1).fillna(0)


df_protein_file.to_csv("protein1.csv")


from tqdm import tqdm

final_df = pd.DataFrame(columns=list(df_protein_file.columns)+["gene_type"]+["startpos.q"]+["endpos.q"]+["seq.r"]+["seq.q"]+["LOV"])
for i, row in tqdm(df_common.iterrows()):
    # Check for the at start
    if df_protein_file[(df_protein_file["Start"]==row["startpos.q"]) & (df_protein_file["Stop"]>row["endpos.q"])].shape[0]>=1:
        tmp_df = df_protein_file[(df_protein_file["Start"]==row["startpos.q"]) & (df_protein_file["Stop"]>row["endpos.q"])].copy(deep=True)
        tmp_df["gene_type"] = "at_start"
        tmp_df["startpos.q"] = row["startpos.q"]
        tmp_df["endpos.q"] = row["endpos.q"]
        tmp_df["seq.r"] = row["seq.r"]
        tmp_df["seq.q"] = row["seq.q"]
        final_df = pd.concat([final_df,tmp_df]).reset_index(drop=True)
        a=a+1
    # check for the at end
    elif df_protein_file[(df_protein_file["Start"]<row["startpos.q"]) & (df_protein_file["Stop"]==row["endpos.q"])].shape[0]>=1:
        tmp_df = df_protein_file[(df_protein_file["Start"]<row["startpos.q"]) & (df_protein_file["Stop"]==row["endpos.q"])].copy(deep=True)
        tmp_df["gene_type"] = "at_end"
        tmp_df["startpos.q"] = row["startpos.q"]
        tmp_df["endpos.q"] = row["endpos.q"]
        tmp_df["seq.r"] = row["seq.r"]
        tmp_df["seq.q"] = row["seq.q"]
        final_df = pd.concat([final_df,tmp_df]).reset_index(drop=True)
        b=b+1
    # Check for the within gene
    elif df_protein_file[(df_protein_file["Start"]<row["startpos.q"]) & (df_protein_file["Stop"]>row["endpos.q"])].shape[0]>=1:
        tmp_df = df_protein_file[(df_protein_file["Start"]<row["startpos.q"]) & (df_protein_file["Stop"]>row["endpos.q"])].copy(deep=True)
        tmp_df["gene_type"] = "within_gene"
        tmp_df["startpos.q"] = row["startpos.q"]
        tmp_df["endpos.q"] = row["endpos.q"]
        tmp_df["seq.r"] = row["seq.r"]
        tmp_df["seq.q"] = row["seq.q"]
        final_df = pd.concat([final_df,tmp_df]).reset_index(drop=True)
        c=c+1
    # Check for the gene_start_with variation end
    elif df_protein_file[(df_protein_file["Start"]==row["endpos.q"])].shape[0]>=1:
        tmp_df = df_protein_file[(df_protein_file["Start"]==row["endpos.q"])].copy(deep=True)
        tmp_df["gene_type"] = "gene_start_with variation end"
        tmp_df["startpos.q"] = row["startpos.q"]
        tmp_df["endpos.q"] = row["endpos.q"]
        tmp_df["seq.r"] = row["seq.r"]
        tmp_df["seq.q"] = row["seq.q"]
        final_df = pd.concat([final_df,tmp_df]).reset_index(drop=True)
        d=d+1
    # Check for the gene_stop_ with variation start
    elif df_protein_file[(df_protein_file["Stop"]==row["startpos.q"])].shape[0]>=1:
        tmp_df = df_protein_file[(df_protein_file["Stop"]==row["startpos.q"])].copy(deep=True)
        tmp_df["gene_type"] = "gene_stop_ with variation start"
        tmp_df["startpos.q"] = row["startpos.q"]
        tmp_df["endpos.q"] = row["endpos.q"]
        tmp_df["seq.r"] = row["seq.r"]
        tmp_df["seq.q"] = row["seq.q"]
        final_df = pd.concat([final_df,tmp_df]).reset_index(drop=True)
        e=e+1
    # Check for the intergenic gene
    elif df_protein_file[(df_protein_file["Stop"]<row["startpos.q"]) & (df_protein_file["start_of_next_gene"]>row["endpos.q"])].shape[0]>=1:
        tmp_df = df_protein_file[(df_protein_file["Stop"]<row["startpos.q"]) & (df_protein_file["start_of_next_gene"]>row["endpos.q"])].copy(deep=True)
        tmp_df["gene_type"] = "intergenic_gene"
        tmp_df["startpos.q"] = row["startpos.q"]
        tmp_df["endpos.q"] = row["endpos.q"]
        tmp_df["seq.r"] = row["seq.r"]
        tmp_df["seq.q"] = row["seq.q"]
        final_df = pd.concat([final_df,tmp_df]).reset_index(drop=True)
        f=f+1
    else:
        pass


final_df.shape

final_df.drop_duplicates().shape

final_df.to_csv("genesv1.csv")

# non coding region
final_df1 = pd.DataFrame(columns=df_common.columns)
for index, row in tqdm(df_common.iterrows()):
    # Check if 'startpos.q' is present in any row of file2
    if row['startpos.q'] not in final_df['startpos.q'].values:
        # Append the row to the DataFrame if not found
        final_df1 = final_df1._append(row, ignore_index=True)

# Save the DataFrame with rows not found in file2 to a new CSV file
final_df1.to_csv("non_coding1.csv", index=False)

final_df1.shape

final_df1.drop_duplicates().shape

imm = pd.DataFrame
imm=final_df[final_df["keyword_found"]==True]

imm.shape

imm.drop_duplicates().shape

imm.drop_duplicates().to_csv("immv1.csv")

# counting the values
# Create a new DataFrame with unique combinations of 'Locus', 'startpos.q', and 'endpos.q'
locus_counts = final_df.drop_duplicates(varet=['Locus', 'startpos.q', 'endpos.q'])

locus_counts.shape

locus_counts.to_csv("locus_counts1.csv")

locus_counts_imm = imm.drop_duplicates(varet=['Locus', 'startpos.q', 'endpos.q'])

locus_counts_imm.shape

locus_counts_imm.to_csv("locus_counts_imm1.csv")

gene_count=locus_counts.groupby(["gene_type"]).size().reset_index(name="count")

gene_count

gene_count.to_csv("gene_count1.csv")

loc_imm_count=locus_counts_imm.groupby(by=["Locus","gene_type"]).size().reset_index(name="count")
# loc_imm_count.columns=["Locus", "gene_type", "count"]

loc_imm_count.to_csv("immlocus_gene_type_counts1.csv", index=False)

loc_imm_count

# Calculate value counts
count = locus_counts["gene_type"].value_counts().reset_index()
count.columns = ["gene_type", "Count"]
count.loc[len(count.index)]=["immune_gene", len(locus_counts_imm)]
count.loc[len(count.index)]=["non_coding_gene", len(final_df1)]


# Save the result to a CSV file
count.to_csv("varcounts1.csv", index=False)


# extract within gene information
loc_count=pd.read_csv("locus_counts1.csv")

loc_count.shape

locus_count_within_gene=loc_count[loc_count["gene_type"]=="within_gene"]
locus_count_inter_gene=loc_count[loc_count["gene_type"]=="intergenic_gene"]

locus_count_within_gene.shape

locus_count_inter_gene.shape

# Create a new DataFrame with unique combinations of 'Locus', 'startpos.q', and 'endpos.q'
unique_counts = locus_count_within_gene.drop_duplicates(varet=['Locus', 'startpos.q', 'endpos.q'])
counts=unique_counts.groupby(["Locus"]).size().reset_index(name="count")

counts.to_csv("loc_var_counts1.csv")

loc50_var=counts[counts["count"]>=50]

loc50_var.shape

locus_count_within_gene.to_csv("locus_within_gene1.csv")
locus_count_inter_gene.to_csv("locus_inter_gene1.csv")
loc50_var.to_csv("loc50_var1.csv")
