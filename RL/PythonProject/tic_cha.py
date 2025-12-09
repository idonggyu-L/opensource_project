import os
import shutil

tickers =  [
    "600519", "601318", "600036", "601166", "600900", "601899", "600030", "601398", "600276", "601328",
    "601288", "600887", "600000", "601088", "601601", "601668", "600016", "600031", "603019", "600309",
    "601169", "601857", "601919", "600690", "600660", "601012", "601688", "600406", "601766", "600809",
    "600050", "600028", "601988", "601006", "601818", "601225", "600104", "600150", "601628", "601009"
]

t = [
    "600999", "603288", "600760", "601939", "600111", "600089", "600436", "601888", "603993", "600048",
    "600547", "600019", "601390", "600415", "601600", "600893", "600585", "600570", "600958", "603799"]

source_dir = "/home/hail/Desktop/stock/SSE/"
dest_dir = "/home/hail/Desktop/stock/data_ch"

os.makedirs(dest_dir, exist_ok=True)


for ticker in tickers:
    filename = f"{ticker}.SS.csv"
    src_path = os.path.join(source_dir, filename)
    dst_path = os.path.join(dest_dir, filename)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f" Copied: {filename}")
    else:
        print(f"Not found: {filename}")
