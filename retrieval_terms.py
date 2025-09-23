from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os



embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db_pdf = FAISS.load_local(
    "faiss_indexes/pdf_combined",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatOpenAI(model="gpt-4o-mini")

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

dict_prompt = PromptTemplate(
    template="""
    질문과 맥락:
    {input_text}

    규칙:
    - 반드시 맥락에 있는 정보만 사용해서 2~3문장으로 설명할 것
    - 맥락이 비어 있거나 질문과 관련 없는 경우에는 "관련 용어 설명을 찾지 못했습니다."라고 답할 것
    """,
    input_variables=["input_text"]
)

terms_chain = LLMChain(
    llm=llm,
    prompt=dict_prompt,
    memory=memory,
    verbose=True
)

def explain_terms(terms: str, k: int = 15) -> str:
    """경제 용어 설명"""
    docs_and_scores = db_pdf.similarity_search_with_score(terms, k=k)
    if not docs_and_scores:
        return None

    best_doc, _ = docs_and_scores[0]
    context = best_doc.page_content

    # question + context 하나로 합쳐서 전달
    return terms_chain.run(input_text=f"질문: {terms}\nContext: {context}")
