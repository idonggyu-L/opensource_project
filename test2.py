# test_terms.py
from retrieval_terms import explain_terms, memory

if __name__ == "__main__":
    print("=== 첫 질문 ===")
    ans1 = explain_terms("금리")
    print("답변:", ans1)

    print("\n=== 이어지는 질문 (메모리 확인) ===")
    ans2 = explain_terms("수요와 공급에 대해 설명해줘")
    print("답변:", ans2)

    print("\n=== 현재 대화 기록 ===")
    print(memory.load_memory_variables({})["chat_history"])
