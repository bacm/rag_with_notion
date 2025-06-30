from langchain.embeddings import HuggingFaceEmbeddings
from fetch_notion import get_chunks_and_model
from psql_helpers import populate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

PAGE_ID = "" # TODO set PAGE ID

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # TODO check if this is the rigth embedding
chunks = get_chunks_and_model(PAGE_ID)
vectorstore = populate(chunks, embeddings)

# 1. Modèle LLM (Mixtral via Ollama)
llm = ChatOpenAI(
    model="llama-3.1-8b-instruct", # TODO ajust model
    api_key="", # TODO set API KEY from 1password
    base_url="" # TODO set URL BASE from 1password
)

# 2. Mémoire de conversation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# 3. Retriever à partir du vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 4. Chaîne RAG conversationnelle
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)


print("🧠 Chatbot RAG prêt. Pose ta question (ou tape quit() pour quitter)")

while True:
    try:
        query = input("\n❓ Question : ")
        if query.strip().lower() == "quit()":
            print("👋 À bientôt.")
            break
        res = qa_chain.invoke({"question": query})
        print(f"💬 Réponse : {res["answer"]}")
        print(res)

        retrieved_docs = retriever.get_relevant_documents(query)
        print(f"\n🧩 Documents retrouvés pour « {query} » : {len(retrieved_docs)}")
        for d in retrieved_docs:
            print(d.page_content[:300])


    except KeyboardInterrupt:
        print("\n👋 Interruption. À bientôt.")
        break