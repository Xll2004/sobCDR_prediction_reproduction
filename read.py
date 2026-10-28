from pathlib import Path
import pandas as pd

parquet_file = Path(
    "/Users/liangxuelin/Desktop/NACC/investigator_nacc75.parquet"
)
csv_file = parquet_file.with_suffix(".csv")

df = pd.read_parquet(parquet_file, engine="pyarrow")

print("All columns:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")

print(f"\nData size: {df.shape[0]} rows × {df.shape[1]} columns")

df.to_csv(csv_file, index=False, encoding="utf-8-sig")

print(f"\nCSV saved to: {csv_file}")