from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from finrag.services.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_markdown(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_markdown(markdown_content: str):
    headers_to_split_on = [
        ("#", "Titre_1"),
        ("##", "Titre_2"),
        ("###", "Titre_3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    md_chunks = md_splitter.split_text(markdown_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    final_chunks = text_splitter.split_documents(md_chunks)

    return final_chunks
