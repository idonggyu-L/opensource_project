import os
import shutil

tickers = [
    "NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "TSLA", "JPM", "LLY", "V",
    "XOM", "ORCL", "MA", "WMT", "JNJ", "HD", "ABBV", "PG", "BAC", "GE",
    "UNH", "CVX", "WFC", "IBM", "AMD", "PM", "KO", "GS", "CRM", "ABT",
    "RTX", "CAT", "MCD", "DIS", "T", "MRK", "NOW", "MS", "C","AXP"]

a = [ "VZ", "TMO", "BA", "BLK", "SCHW", "TJX", "NEE",
"SPGI", "APH", "ACN", "ANET", "BSX", "LOW", "COF", "PGR", "ETN", "UNP", "PFE", "BX", "SYK", "COP", "MDT",
"DHR", "WELL", "DE", "MO", "PLD", "CB", "SO", "LMT", "MMC", "CVS", "ICE", "PH", "DUK", "MCK", "NEM", "TT",
"KKR", "AMT", "BMY", "GD", "RCL", "NKE", "MMM", "WM", "PNC", "NOC", "SHW", "WMB", "AJG", "BK", "USB",
"AON", "CI", "MSI", "MCO", "TDG", "EMR", "ELV"]

source_dir = "/home/hail/Desktop/stock/NYSE/"
dest_dir = "/home/hail/Desktop/stock/data_us"

os.makedirs(dest_dir, exist_ok=True)


for ticker in tickers:
    filename = f"{ticker}.csv"
    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(dest_dir, filename)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f" Copied: {filename}")
    else:
        print(f"Not found: {filename}")
