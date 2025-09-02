import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document



# 2. Load multiple CSV files
files = [
    "naver_news_finance.csv"#,
    #"naver_news_securities.csv",
    #"naver_news_industry.csv",
    #"naver_news_realestate.csv",
    #"naver_news_global.csv",
    #"naver_news_life.csv"
]

docs = []
for f in files:
    df = pd.read_csv(f)
    for _, row in df.iterrows():
        # each row becomes a Document for LangChain
        docs.append(
            Document(
                page_content=row["content"],
                metadata={"title": row["title"], "link": row["link"]}
            )
        )

# 3. Split text into chunks (optional but recommended for long articles)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# 4. Create embeddings + FAISS vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(split_docs, embeddings)

# 5. LLM + Memory
llm = ChatOpenAI(model="gpt-4o-mini")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 6. QA chain with retrieval
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
    chain_type="stuff",
    memory=memory,
    verbose=True
)

# 7. Example conversation
print("Q:", "금 관련 정보")
aa = qa.invoke("금 관련 정보")
print("A:",aa['result'] )

