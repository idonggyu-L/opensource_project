import time
import threading
from crawler import run_crawler       # 크롤러
from build_db import build_news_db    # DB 빌더

def update_news(csv_dir="/home/hail/RAG/data", interval=3600, max_pages=5):
    """주기적으로 뉴스 크롤링 → DB 업데이트"""
    def loop():
        while True:
            print("🌀 뉴스 크롤링 + DB 업데이트 시작")
            try:
                # 1) 최신 뉴스 크롤링 → CSV 저장
                run_crawler(max_pages=max_pages, output_dir=csv_dir)

                # 2) CSV 기반으로 뉴스 벡터 DB 재구축
                build_news_db(csv_dir)
                print("✅ 뉴스 DB 업데이트 완료")
            except Exception as e:
                print(f"❌ 업데이트 실패: {e}")

            # interval 만큼 대기
            time.sleep(interval)

    # 백그라운드 스레드에서 실행
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
