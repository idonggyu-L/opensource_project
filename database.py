# build_index.py
import pandas as pd
import faiss
import numpy as np
from openai import OpenAI



# Load CSV (title, link, content)
df = pd.read_csv("naver_news_finance.csv")

# Create embeddings
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"   # or "text-embedding-3-large"
    )
    return response.data[0].embedding

# Generate embeddings for all articles
embeddings = [get_embedding(row["content"]) for _, row in df.iterrows()]
embeddings = np.array(embeddings, dtype="float32")

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index and dataframe
faiss.write_index(index, "naver_news.index")
df.to_csv("naver_news_finance_with_index.csv", index=False, encoding="utf-8-sig")

print(f"Added {len(embeddings)} articles to FAISS index and saved to disk.")