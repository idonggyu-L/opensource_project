import os
import shutil

tickers = [
    "005930", "000660", "105560", "012450", "035420", "055550", "005380",
    "068270", "034020", "000270", "086790", "035720", "005490", "028260",
    "009540", "012330", "316140", "402340", "064350", "207940", "010140",
    "000810", "032830", "051910", "267260", "033780", "006400", "373220",
    "042660", "015760", "030200", "009150", "329180", "138040", "259960",
    "066570", "017670", "323410", "047810", "096770"
]

ti_ = ["034730", "000100", "011200", "267250", "003550"]

t = ["079550", "005830", "278470", "003230"]

tt = ["018260", "086280"]

t_ = ["006800"]

source_dir = "/home/hail/Desktop/stock/KOSPI/"
dest_dir = "/home/hail/Desktop/stock/data_"

os.makedirs(dest_dir, exist_ok=True)


for ticker in t_:
    filename = f"{ticker}.KS.csv"
    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(dest_dir, filename)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f" Copied: {filename}")
    else:
        print(f"Not found: {filename}")
