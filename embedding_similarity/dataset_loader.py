import pandas as pd


def load_dataset(path="common/synthetic_ecommerce_data.csv"):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df["text"].tolist(), df["intent"].tolist()
