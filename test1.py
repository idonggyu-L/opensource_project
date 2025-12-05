# test_news.py
from retrieval_news import search_news
import os
import locale

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LC_ALL"] = "C.UTF-8"
os.environ["LANG"] = "C.UTF-8"
locale.setlocale(locale.LC_ALL, "C.UTF-8")


def sanitize_query(text: str) -> str:
    return text.encode('utf-8', 'ignore').decode('utf-8')

if __name__ == "__main__":
    query1 = sanitize_query("코스피 관련 뉴스 요약해줘")
    ans1 = search_news(query1)
    print("답변:", ans1)

