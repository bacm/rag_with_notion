from langchain.vectorstores.pgvector import PGVector

CONNECTION_STRING = "postgresql+psycopg2://postgres:yourpassword@localhost:5432/notion"

def populate(chunks, embeddings):
    return PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="notion_docs",
        connection_string=CONNECTION_STRING,
    )