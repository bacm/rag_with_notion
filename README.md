# RAG with Notion

This repository provides a foundation for a Retrieval-Augmented Generation (RAG) system connected to Notion.  
It is used as a starting point for an internal **training session** at **Manty**.

## 🔍 Overview

This project enables querying and indexing content from Notion pages using a RAG architecture, combining:

- **Notion API**: to fetch structured content from Notion databases and pages.
- **LangChain**: to process and split text into vectorizable chunks.
- **Vector store (PGVector)**: to store and retrieve semantically relevant documents.
- **LLM (OpenAI / local models)**: to answer questions grounded on retrieved Notion content.

## 🧱 Features

- Fetch Notion blocks and convert them into LangChain-compatible documents.
- Support for recursive Notion page traversal.
- Chunking of documents using `RecursiveCharacterTextSplitter`.
- Embedding and vector store integration (PGVector).
- Basic RAG agent ready to be extended for more complex workflows.
