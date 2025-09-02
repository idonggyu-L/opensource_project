# query_rag.py
import pandas as pd
import faiss
import numpy as np
from openai import OpenAI



# Load FAISS index + dataframe
index = faiss.read_index("naver_news.index")
df = pd.read_csv("naver_news_finance_with_index.csv")


# Embedding function (same as build_index.py)
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


# Search function
def search(query, top_k=3):
    query_vec = np.array(get_embedding(query), dtype="float32").reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i in indices[0]:
        results.append({
            "title": df.iloc[i]["title"],
            "link": df.iloc[i]["link"],
            "content": df.iloc[i]["content"]
        })
    return results


# RAG function (GPT)
def answer_with_rag(query):
    context_docs = search(query, top_k=3)
    context_text = "\n\n".join([doc["content"] for doc in context_docs])

    prompt = f"""
    User query: {query}

    Relevant news:
    {context_text}

    Answer in Korean, summarize concisely based on the news above.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# Example usage
if __name__ == "__main__":
    query = "금"
    print("Query:", query)
    print(answer_with_rag(query))
