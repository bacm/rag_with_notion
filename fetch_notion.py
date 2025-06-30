from notion_client import Client
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import requests

NOTION_TOKEN = "" # TODO set token
notion = Client(auth=NOTION_TOKEN)

# Récupère les blocs de la page
def fetch_all_blocks_recursive(block_id, depth=0):
    results = []
    cursor = None
    while True:
        response = notion.blocks.children.list(block_id=block_id, start_cursor=cursor)
        blocks = response["results"]
        for block in blocks:
            block["depth"] = depth  # pour info
            results.append(block)

            # Certains types peuvent avoir des enfants
            if block.get("has_children"):
                child_id = block["id"]
                children = fetch_all_blocks_recursive(child_id, depth=depth + 1)
                results.extend(children)

        cursor = response.get("next_cursor")
        if not cursor:
            break
    return results

# Transforme les blocs en Documents LangChain
def parse_blocks_to_documents(blocks, page_title):
    documents = []
    current_section = ""

    for block in blocks:
        block_type = block.get("type")
        rich_text = block.get(block_type, {}).get("rich_text", [])
        text = "".join(rt.get("plain_text", "") for rt in rich_text)

        if block_type.startswith("heading"):
            current_section = text
            continue
        elif block_type == "paragraph" and text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"section": current_section, "page_title": page_title}
            ))
        elif block_type == "image":
            caption = block.get("image", {}).get("caption", [])
            caption_text = "".join(rt.get("plain_text", "") for rt in caption)
            documents.append(Document(
                page_content=f"[IMAGE: {caption_text}]",
                metadata={"section": current_section, "page_title": page_title}
            ))

    return documents

# Récupère le titre de la page
def get_page_title(page_id):
    page = notion.pages.retrieve(page_id=page_id)
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            return "".join([t["plain_text"] for t in prop["title"]])
    return "Page sans titre"

def get_chunks_and_model(page_id): 
    blocks = fetch_all_blocks_recursive(page_id)
    print(f"[DEBUG] Nombre total de blocs (récursifs) : {len(blocks)}")

    page_title = get_page_title(page_id)
    docs = parse_blocks_to_documents(blocks, page_title)
    print(f"[DEBUG] Nombre de documents utiles : {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    return chunks
