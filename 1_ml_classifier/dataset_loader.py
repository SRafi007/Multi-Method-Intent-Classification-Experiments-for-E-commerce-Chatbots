import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path="data/synthetic_ecommerce_data.csv", test_size=0.2):
    df = pd.read_csv(path)
    df.dropna(inplace=True)

    X = df["text"].astype(str).tolist()
    y = df["intent"].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test
