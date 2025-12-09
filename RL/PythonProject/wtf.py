import os
import pandas as pd


def merge_kospi_data(data_dir: str, save_path: str) -> pd.DataFrame:
    """Concatenate multiple KOSPI CSVs into a long format dataset."""

    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    dfs = []

    for file in all_files:
        file_path = os.path.join(data_dir, file)
        tic = file.replace(".csv", "")   # Extract ticker

        df = pd.read_csv(file_path)

        # Rename columns to match target schema
        df = df.rename(columns={
            "Date": "datadate",
            "Adj Close": "prccd",
            "Close": "prcod",
            "High": "prchd",
            "Low": "prcld",
            "Volume": "cshtrd"
        })

        # Convert date to datetime
        df["datadate"] = pd.to_datetime(df["datadate"])

        # Adjustment index (set to 1.0 for KRX data)
        df["ajexdi"] = 1.0

        # Add ticker
        df["tic"] = tic

        # Reorder columns
        df = df[["datadate", "tic", "prccd", "ajexdi", "prcod", "prchd", "prcld", "cshtrd"]]

        # Filter date range
        df = df[(df["datadate"] >= "2016-01-01") & (df["datadate"] <= "2025-03-31")]

        dfs.append(df)

    # Concatenate without aligning tickers
    merged_df = pd.concat(dfs, ignore_index=True)

    # Sort so that rows for each ticker stay together and ordered by date
    merged_df = merged_df.sort_values(by=["tic", "datadate"]).reset_index(drop=True)

    # Save
    merged_df.to_csv(save_path, index=False)

    return merged_df

def concat_kospi_data_(data_dir: str, save_path: str) -> pd.DataFrame:
    """Concatenate KOSPI CSVs (long format) and standardize column names."""

    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    dfs = []

    for file in all_files:
        file_path = os.path.join(data_dir, file)
        tic = file.replace(".csv", "")

        df = pd.read_csv(file_path)

        # ? Ä®·³¸í Ç¥ÁØÈ­ (FinRL È¯°æ ±âÁØ)
        df = df.rename(columns={
            "Date": "datadate",
            "Adj Close": "adjcp",
            "Close": "close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Volume": "volume"
        })

        # ³¯Â¥ Ã³¸®
        df["datadate"] = pd.to_datetime(df["datadate"])

        # Á¾¸ñÄÚµå Ãß°¡
        df["tic"] = tic

        # ¼ø¼­ Á¤¸®
        df = df[["datadate", "tic", "adjcp", "open", "high", "low", "close", "volume"]]

        dfs.append(df)

    # ´Ü¼ø concat (¼¯Áö ¾ÊÀ½)
    merged_df = pd.concat(dfs, ignore_index=True)

    # ÀúÀå
    merged_df.to_csv(save_path, index=False)
    return merged_df



if __name__ == "__main__":
    DATA_DIR = "/home/hail/Desktop/stock/data_ch/"   # Replace with your folder path
    SAVE_PATH = "/home/hail/Desktop/stock/new/ch.csv"

    merged_df = concat_kospi_data_(DATA_DIR, SAVE_PATH)

    print("Final dataset shape:", merged_df.shape)
    print("Saved to:", SAVE_PATH)
