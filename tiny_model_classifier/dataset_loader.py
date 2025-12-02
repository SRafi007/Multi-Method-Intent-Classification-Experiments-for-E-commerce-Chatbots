# scripts/dataset_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List

def load_csv(path: str = "common/synthetic_ecommerce_data.csv", test_size: float = 0.15, val_size: float = 0.10, random_state: int = 42
            ) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    Expects CSV with columns: text,intent
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "intent"])
    # Filter out classes with fewer than 2 samples
    v_counts = df["intent"].value_counts()
    df = df[df["intent"].isin(v_counts[v_counts > 1].index)]
    X = df["text"].astype(str).tolist()
    y = df["intent"].astype(str).tolist()

    # First split train+val vs test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Then split train vs val
    val_ratio_of_tmp = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio_of_tmp, random_state=random_state, stratify=y_tmp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
