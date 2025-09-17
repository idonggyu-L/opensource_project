# test_news.py
from retrieval_news import search_news, memory

if __name__ == "__main__":
    print("=== 첫 질문 ===")
    ans1 = search_news("최근 금리 인상 관련 뉴스 알려줘")
    print("답변:", ans1)

    print("\n=== 이어지는 질문 (메모리 확인) ===")
    ans2 = search_news("그 뉴스에서 한국은행 얘기도 있었어?")
    print("답변:", ans2)

    print("\n=== 현재 대화 기록 ===")
    print(memory.load_memory_variables({})["chat_history"])
