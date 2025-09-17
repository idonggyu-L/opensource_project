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
    - 반드시 맥락에 있는 내용만 사용해서 3~4문장으로 요약해라.
    - 맥락이 비어 있으면 "관련 뉴스를 찾지 못했습니다."라고 답변해라.
    """,
    input_variables=["input_text"]   # ✅ 단일 입력
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

def search_news(query: str, k: int = 3) -> str:
    docs_and_scores = db_news.similarity_search_with_score(query, k=k)
    if not docs_and_scores:
        return "관련 뉴스를 찾지 못했습니다."

    best_doc, _ = docs_and_scores[0]
    context = best_doc.page_content

    # ✅ question + context 합쳐서 하나의 input_text로 전달
    return news_chain.run(input_text=f"질문: {query}\nContext: {context}")

