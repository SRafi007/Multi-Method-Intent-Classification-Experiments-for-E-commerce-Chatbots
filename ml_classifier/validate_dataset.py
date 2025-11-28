import pandas as pd
from collections import Counter

def validate_dataset(path="data/intents.csv", auto_fix=False):
    print("\n=== DATASET VALIDATION ===\n")
    
    df = pd.read_csv(path)

    # 1. Basic checks
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'intent' columns.")

    # 2. Drop empty rows
    empty_rows = df[df["text"].isna() | df["intent"].isna()]
    if len(empty_rows) > 0:
        print(f" Found {len(empty_rows)} empty rows.")
    
    df = df.dropna()

    # 3. Show class counts
    counts = Counter(df["intent"])
    print("\nIntent counts:")
    for intent, count in counts.items():
        print(f"  {intent}: {count}")

    # 4. Detect classes with too few samples
    too_small = [label for label, cnt in counts.items() if cnt < 2]
    if too_small:
        print("\n Classes with too few samples (<2):")
        for label in too_small:
            print("  -", label)
    else:
        print("\nAll classes have enough samples.")

    # 5. Detect duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicates: {duplicates}")

    if duplicates > 0:
        print(" You should remove duplicates.")

    # 6. Optional: Auto-fix small classes
    if auto_fix and too_small:
        print("\nAuto-fix enabled → Generating synthetic samples...")
        df = _fix_small_classes(df, too_small)
        df.to_csv(path.replace(".csv", "_fixed.csv"), index=False)
        print("Saved fixed dataset to:", path.replace(".csv", "_fixed.csv"))

    print("\n=== VALIDATION COMPLETE ===\n")
    return df


# Optional synthetic sample generator
def _fix_small_classes(df, small_classes):
    from random import choice

    new_rows = []

    for intent in small_classes:
        txt = df[df["intent"] == intent].iloc[0]["text"]
        new_text = txt + " (similar query)"
        new_rows.append({"text": new_text, "intent": intent})

    print(f"Generated {len(new_rows)} synthetic samples.")
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


if __name__ == "__main__":
    validate_dataset(auto_fix=False)
