from retrieval_news import search_news
from retrieval_terms import explain_terms
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import json



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    user_input: str

llm = ChatOpenAI(model="gpt-4o-mini")

# 용어 추출 체인
keyword_prompt = PromptTemplate(
    template="""
    아래 뉴스 요약에서 경제/금융 관련 핵심 용어 3~5개만 뽑아서
    반드시 JSON 배열 형식으로만 출력해라.
    예시: ["기준금리","FOMC","금리"]

    뉴스 요약:
    {context}
    """,
    input_variables=["context"]
)
keyword_chain = LLMChain(llm=llm, prompt=keyword_prompt, verbose=True)

def extract_terms_with_llm(context: str) -> list[str]:
    raw_output = keyword_chain.run(context=context)
    try:
        terms = json.loads(raw_output)
        if isinstance(terms, list):
            return [t.strip() for t in terms]
    except Exception as e:
        print("⚠️ JSON parse 실패:", e, "| raw_output:", raw_output)
    return []

def route_query(user_input: str):
    # 1) 뉴스 검색 + 요약
    news_result = search_news(user_input, k=5)
    news_summary = news_result["summary"]
    news_links = news_result["links"]

    # 2) 뉴스 요약 기반 용어 추출
    keywords = extract_terms_with_llm(news_summary)

    # 3) 용어 설명 (사전 DB 참고)
    terms_answer = {}
    for term in keywords:
        explanation = explain_terms(term)
        if explanation and "찾지 못했습니다" not in explanation:
            terms_answer[term] = explanation

    # 4) 최종 응답 구조
    response = {
        "뉴스 요약": news_summary,
        "관련 용어": terms_answer,
        "출처": news_links
    }
    return response

@app.post("/query")
def query_api(request: QueryRequest):
    return route_query(request.user_input)
