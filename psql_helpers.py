from langchain.vectorstores.pgvector import PGVector
from urllib.parse import quote_plus

# TODO set postgres credentials and settings
user = ""
port = ""
raw_password = ""
ip = ""

encoded_password = quote_plus(raw_password) # handles @ in password

CONNECTION_STRING = f"postgresql+psycopg2://{user}:{encoded_password}@{ip}:{port}/notion"

def populate(chunks, embeddings):
    print(f"[DEBUG] Nombre de chunks récupérés : {len(chunks)}")
    return PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="notion_docs",
        connection_string=CONNECTION_STRING,
    )