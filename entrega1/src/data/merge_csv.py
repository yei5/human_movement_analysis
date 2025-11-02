import os
import pandas as pd

def merge_csvs(csv_dir, output_path):
    all_data = []
    for fname in os.listdir(csv_dir):
        if fname.endswith(".csv") and fname != "merged_dataset.csv":
            activity = fname.split("-")[0] if "-" in fname else os.path.splitext(fname)[0]
            df = pd.read_csv(os.path.join(csv_dir, fname))
            df["activity"] = activity
            df["source_file"] = fname
            all_data.append(df)
    
    merged = pd.concat(all_data, ignore_index=True)
    merged.to_csv(output_path, index=False)
    print(f"Merged dataset saved to {output_path} with {len(merged)} frames")

if __name__ == "__main__":
    merge_csvs(
        "entrega1/data/processed/landmarks/",
        "entrega1/data/processed/landmarks/merged_dataset.csv"
    )
