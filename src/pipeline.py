import pandas as pd

def load_gold_data(path="data/gold/final_abuse_detection_dataset.parquet"):
    return pd.read_parquet(path)

def clean_text(df):
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]
    return df
