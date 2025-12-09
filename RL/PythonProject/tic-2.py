import os
import shutil

tickers = [
    "9983", "9984", "6857", "8035", "9433", "6762", "6098", "4063", "9766", "4543",
    "6954", "6758", "4519", "6367", "6988", "6971", "7832", "7203", "7974", "5803",
    "8015", "9735", "4568", "4901", "8058", "7267", "7741", "8766", "6902", "8001",
    "6146", "4503", "2802", "7269", "4704", "4578", "4507", "7733", "8031", "9843"
]

t = ["3659", "4452", "5108", "7751", "7453"]

source_dir = "/home/hail/Desktop/stock/TSE/"
dest_dir = "/home/hail/Desktop/stock/data_jp"

os.makedirs(dest_dir, exist_ok=True)


for ticker in tickers:
    filename = f"{ticker}.T.csv"
    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(dest_dir, filename)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f" Copied: {filename}")
    else:
        print(f"Not found: {filename}")
