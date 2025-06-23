
from langchain.vectorstores import Chroma

def populate(chunks, embeddings):
    # Crée et persiste le vecteur
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="notion_chroma")
    vectorstore.persist()

    vectorstore = Chroma(persist_directory="notion_chroma", embedding_function=embeddings)
    return vectorstore