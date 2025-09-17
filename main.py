from retrieval_news import search_news
from retrieval_terms import explain_terms
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI
from pydantic import BaseModel
from retrieval_news import search_news, db_news
from retrieval_terms import explain_terms
from langchain.memory import ConversationBufferMemory
from fastapi.middleware.cors import CORSMiddleware


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

def route_query(user_input: str):
    if "용어" in user_input or "자세히 설명" in user_input:
        return {
            "type": "terms",
            "terms_answer": explain_terms(user_input)
        }
    else:
        # 뉴스 검색
        docs_and_scores = db_news.similarity_search_with_score(user_input, k=1)
        if not docs_and_scores:
            return {"type": "news", "news_answer": "관련 뉴스를 찾지 못했습니다."}

        best_doc, _ = docs_and_scores[0]
        news_context = best_doc.page_content

        # 뉴스 요약
        news_answer = search_news(user_input)

        # 뉴스 본문(context) 기반 용어 설명
        terms_answer = explain_terms(news_context)

        return {
            "type": "news",
            "news_answer": news_answer,
            "terms_answer": terms_answer
        }

@app.post("/query")
def query_api(request: QueryRequest):
    return route_query(request.user_input)


