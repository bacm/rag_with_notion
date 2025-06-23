import requests

# assume que mixtral tourne en local sur ollaman
def rag_with_ollama_mixtral(prompt: str):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mixtral",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# Exemple de RAG minimal
def ask_question(query, vectorstore):
    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
Voici des documents extraits de Notion :

{context}

Réponds à la question suivante en te basant uniquement sur ces documents :
{query}
"""
    return rag_with_ollama_mixtral(prompt)
