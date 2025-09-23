import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain.text_splitter import TokenTextSplitter




def safe_faiss_from_documents(docs, embeddings, batch_size=1000):
    """Create FAISS index safely by splitting into smaller batches."""
    db = None
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        batch_db = FAISS.from_documents(batch, embeddings)
        if db is None:
            db = batch_db
        else:
            db.merge_from(batch_db)
    return db


def build_news_db(csv_dir: str, output_dir: str = "faiss_indexes"):
    """뉴스 기사 DB 생성"""
    os.makedirs(output_dir, exist_ok=True)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    all_news_docs = []
    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            loader = CSVLoader(file_path=os.path.join(csv_dir, file), source_column="content")
            docs = loader.load()
            all_news_docs.extend(docs)

    # 뉴스 기사는 chunk 크게 (기사 단위 유지)
    splitter = TokenTextSplitter(chunk_size=1200, chunk_overlap=100)
    all_news_docs = splitter.split_documents(all_news_docs)

    db_news = safe_faiss_from_documents(all_news_docs, embeddings, batch_size=500)
    db_news.save_local(os.path.join(output_dir, "news_combined"))
    print(f"✅ Saved News DB: {len(all_news_docs)} chunks")

    return db_news


def build_pdf_db(pdf_dir: str, output_dir: str = "faiss_indexes"):
    """경제 용어/교육자료 PDF DB 생성"""
    os.makedirs(output_dir, exist_ok=True)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    level_map = {
        "dictionary": "dict_eco.pdf",
        "elementary": "low.pdf",
        "middle": "middle.pdf",
        "high": "high.pdf",
    }

    all_pdf_docs = []
    for level, filename in level_map.items():
        file_path = os.path.join(pdf_dir, filename)
        if not os.path.exists(file_path):
            print(f"Skipped {filename} (not found)")
            continue

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for d in docs:
            d.metadata["level"] = level
        all_pdf_docs.extend(docs)

    # PDF는 정의 단위로 작게
    splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=50)
    all_pdf_docs = splitter.split_documents(all_pdf_docs)

    db_pdf = safe_faiss_from_documents(all_pdf_docs, embeddings, batch_size=500)
    db_pdf.save_local(os.path.join(output_dir, "pdf_combined"))
    print(f"✅ Saved PDF DB: {len(all_pdf_docs)} chunks (with level metadata)")

    return db_pdf


if __name__ == "__main__":
    # 필요할 때만 골라 실행
    csv_dir = "/home/hail/RAG/data"
    pdf_dir = "/Users/idong-gyu/RAG_clean/data_"

    # 뉴스 DB 생성
    #build_news_db(csv_dir)

    # PDF DB 생성
    build_pdf_db(pdf_dir)


