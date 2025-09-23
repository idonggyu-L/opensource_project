from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os



embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db_news = FAISS.load_local(
    "faiss_indexes/news_combined",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOpenAI(model="gpt-4o-mini")

news_prompt = PromptTemplate(
    template="""
    질문과 맥락:
    {input_text}

    규칙:
    - 반드시 맥락 전체를 고려해 하나의 최종 요약을 작성할 것.
    - 요약은 반드시 최대 5줄 이하로 제한할 것.
    - 기사별 요약이 아니라 전체 내용을 통합해서 핵심만 정리할 것.
    - 불필요한 반복, 중복 문장은 절대 포함하지 말 것.
    - 맥락이 비어 있으면 "관련 뉴스를 찾지 못했습니다."라고 답변해라.
    """,
    input_variables=["input_text"]
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

news_chain = LLMChain(
    llm=llm,
    prompt=news_prompt,
    memory=memory,
    verbose=True
)

def search_news(query: str, k: int = 1) -> dict:
    """뉴스 검색 후, 통합 요약 + 링크 반환"""
    docs_and_scores = db_news.similarity_search_with_score(query, k=k)
    if not docs_and_scores:
        return {"summary": "관련 뉴스를 찾지 못했습니다.", "links": []}

    # 뉴스 context 합치기
    combined_context = ""
    links = []
    for doc, _ in docs_and_scores:
        combined_context += doc.page_content + "\n\n"
        if "url" in doc.metadata:
            links.append(doc.metadata["url"])
        elif "source" in doc.metadata:
            links.append(doc.metadata["source"])

    # 길이 제한 (예: 3000자)
    combined_context = combined_context[:3000]

    # 요약 실행
    summary = news_chain.run(input_text=f"질문: {query}\nContext: {combined_context}")

    # ✅ 5줄 이하 강제
    summary_lines = [line.strip() for line in summary.split("\n") if line.strip()]
    summary = "\n".join(summary_lines[:5])

    return {
        "summary": summary,
        "links": links
    }
