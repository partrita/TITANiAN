import pandas as pd
import sys
from pathlib import Path


def clean_and_split_str(s):
    """Clean and split a string for allele processing."""
    if pd.isna(s):
        return s
    s = str(s)
    if s.startswith("HLA"):
        s = s[4:]
    if "_" in s:
        s = s.replace("_", "")
    if "-" in s:
        s = s.replace("-", "")
    return s


def process_allele_entry(s):
    """Process allele entry with special delimiters and return a cleaned string."""
    if pd.isna(s):
        return s
    s = str(s)
    if s.startswith("HLA"):
        s = s[4:]
    b = "STRANGER"
    if "/" in s:
        a, b = s.split("/", 1)
    else:
        a = s
    a = a.replace("*", "").replace(":", "")
    b = b.replace("*", "").replace(":", "") if b != "STRANGER" else b
    if b != "STRANGER":
        return f"{a}-{b}"
    return a


def load_mhc_data(class_):
    """Load MHC class data based on the specified class."""
    parent = Path(__file__).resolve(True).parent
    if class_ == "I":
        return pd.read_csv(parent / "MHC_classI_pseudo.csv")
    elif class_ == "II":
        return pd.read_csv(parent / "MHC_classII_pseudo.csv")
    else:
        raise ValueError("MHC class should be 'I' or 'II'")


def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py <class> <input_csv> <output_csv>")
        sys.exit(1)
    class_ = sys.argv[1]
    input_df_path = sys.argv[2]
    output_df_path = sys.argv[3]

    try:
        df1 = load_mhc_data(class_)
        df2 = pd.read_csv(input_df_path)

        df1["allele"] = df1["allele"].apply(clean_and_split_str)
        df2["Allele"] = df2["Allele"].apply(process_allele_entry)

        merged_df = pd.merge(df2, df1, left_on="Allele", right_on="allele", how="left")
        merged_df["allele"].fillna("NONE", inplace=True)
        count_none = len(merged_df[merged_df["allele"] == "NONE"])
        merged_df = merged_df[merged_df["allele"] != "NONE"].drop_duplicates(
            subset=["peptide", "Allele"]
        )

        merged_df.to_csv(output_df_path, index=False)
        print(f'Number of "NONE" entries: {count_none}')

    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
