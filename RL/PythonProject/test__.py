import os
import pandas as pd


def merge_stock_csvs(data_dir: str, save_path: str = None) -> pd.DataFrame:
    """
    Merge multiple stock CSV files into a single DataFrame.
    Align all tickers to the same starting date (latest common date).
    Output format: tic | datadate | adjcp | open | high | low | close | volume
    """
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    dfs = []
    min_dates = []

    for file in all_files:
        tic = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(data_dir, file))

        df = df.rename(columns={
            "Date": "datadate",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjcp",
            "Volume": "volume"
        })

        df["datadate"] = pd.to_datetime(df["datadate"]).dt.strftime("%Y%m%d").astype(int)
        df["tic"] = tic
        df = df[["tic", "datadate", "adjcp", "open", "high", "low", "close", "volume"]]

        dfs.append(df)

        # save first date for this stock
        min_dates.append(df["datadate"].min())

    # 1. Find the latest among all min dates (common start date)
    common_start = max(min_dates)
    print(f"Common start date (latest across all tickers): {common_start}")

    # 2. Filter each df so only rows >= common_start remain
    dfs = [df[df["datadate"] >= common_start] for df in dfs]

    # 3. Concatenate and sort
    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.sort_values(by=["datadate", "tic"]).reset_index(drop=True)

    if save_path:
        final_df.to_csv(save_path, index=False)

    return final_df


if __name__ == "__main__":
    data_dir = "/home/hail/Desktop/stock/KOSPI/"
    save_path = "/home/hail/Desktop/stock/new/merged_kospi_aligned.csv"

    merged_df = merge_stock_csvs(data_dir, save_path)
    print(merged_df.head(50))
    print(f"Merged data saved to {save_path}")
