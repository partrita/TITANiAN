import pandas as pd
import sys
from pathlib import Path

parent = Path(__file__).resolve(True).parent


# Function to modify the "MHC Restriction - Name" entries
def modify_entry(a):
    if "_" in a:
        a = a.split("_")[0] + a.split("_")[1]
    if a.startswith("HLA"):
        a = a[4:]
    if "-" in a:
        a = a.split("-")[0] + a.split("-")[1]
    return a

def modify_entry_2(a):
    if a.startswith("HLA"):
        a = a[4:]
    b="STRANGER"
    if "/" in a:
        list_ = a.split("/")
        a = list_[0]
        b = list_[1]
    if "*" in a:
        a= a.split("*")[0] + a.split("*")[1]
    if ":" in a:
        a = a.split(":")[0]+ a.split(":")[1]
    if "*" in b:
        b = b.split("*")[0] + b.split("*")[1]
    if ":" in b:
        b = b.split(":")[0] + b.split(":")[1]
    if b != "STRANGER":
        a = a+"-"+b
    return a
class_ = sys.argv[1]
input_df = sys.argv[2]
output_df = sys.argv[3]


# Load the CSV files into DataFrames
if class_ == "I":
    df1 = pd.read_csv(parent / 'MHC_classI_pseudo.csv')
elif class_ == "II":
    df1 = pd.read_csv(parent / 'MHC_classII_pseudo.csv')
else:
    print("you should specify MHC class: I or II")


df2 = pd.read_csv(str(input_df))

# Modify "MHC Restriction - Name" in df2
df1['allele'] = df1['allele'].apply(lambda a:modify_entry(a))
df2['Allele'] = df2['Allele'].apply(lambda a: modify_entry_2(a))
#print(df1['allele'].tolist())
#print(df2['Allele'].tolist())
# Merge the DataFrames based on the matching columns
merged_df = pd.merge(df2, df1, left_on='Allele', right_on='allele', how='left')

# Fill 'NONE' for the non-matching rows and count them
merged_df['allele'].fillna('NONE', inplace=True)
count_none = len(merged_df[merged_df['allele'] == 'NONE'])
merged_df = merged_df[merged_df['allele'] != 'NONE'].drop_duplicates(subset=['peptide', 'Allele'])
# Save the merged DataFrame to a new CSV file
merged_df.to_csv(str(output_df), index=False)

print(f'Number of "NONE" entries: {count_none}')
