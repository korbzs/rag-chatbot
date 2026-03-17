import os
import sys
import re
from langchain_ollama import OllamaEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings

import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from config import CHROME_DB_PATH, EMBED_MODEL_NAME

def clean_markdown_text(text: str) -> str:
    # " `ű` ", " [ő] "
    text = re.sub(r"['\"`\[]\s*([őűŐŰ])\s*['\"`\]]", r"\1", text)
    
    # "országgyű lési" -> "országgyűlési"
    text = re.sub(r'([a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ])\s+([őűŐŰ])\s+([a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ])', r'\1\2\3', text)
    text = re.sub(r'([a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ])\s+([őűŐŰ])', r'\1\2', text)
    text = re.sub(r'([őűŐŰ])\s+([a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ])', r'\1\2', text)

    stop_words = r'\b(a|az|és|is|be|ki|fel|le|meg)\b'
    def fragment_joiner(match):
        w1, w2 = match.group(1), match.group(2)
        if re.match(stop_words, w1, re.IGNORECASE) or re.match(stop_words, w2, re.IGNORECASE):
            return f"{w1} {w2}"
        return f"{w1}{w2}"

    text = re.sub(r'(\w+)\s+([a-z]{1,2})\b', fragment_joiner, text)
    
    text = re.sub(r'meg\s+rzése', 'megőrzése', text)
    text = re.sub(r'm\s+veletek', 'műveletek', text)
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text) # page numbers
    text = re.sub(r'\n{3,}', '\n\n', text)

    # exact matches better
    text = text.replace("[`ű`]", "ű").replace("[`ő`]", "ő").replace("`", "").replace("** ő **", "ő").replace("** ű **", "ű")
    
    return text

def process_pdf_to_markdown(pdf_path: str) -> str:
    print(f"Converting {os.path.basename(pdf_path)} to Markdown (this might take a while)...")
    return pymupdf4llm.to_markdown(pdf_path)

def ingest_documents(data_dir: str):
    absolute_db_path = os.path.join(PROJECT_ROOT, CHROME_DB_PATH.strip("./"))
    md_temp_dir = os.path.join(data_dir, "md_temp")
    os.makedirs(md_temp_dir, exist_ok=True)
    
    # Ilyenkor újracsinálja a markdownokat a pdf-ekből
    force_rerun = "--rerun" in sys.argv
    if force_rerun:
        print("--- RERUN FLAG DETECTED: Forcing PDF to Markdown conversion ---")
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )

    all_chunks = []

    for root, _, files in os.walk(data_dir):
        if "md_temp" in root:
            continue
            
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                
                language = "unknown"
                if "/en/" in file_path.lower() or "\\en\\" in file_path.lower():
                    language = "en"
                elif "/hu/" in file_path.lower() or "\\hu\\" in file_path.lower():
                    language = "hu"

                try:
                    md_filename = file.replace(".pdf", ".md")
                    md_filepath = os.path.join(md_temp_dir, md_filename)
                    
                    if os.path.exists(md_filepath) and not force_rerun:
                        print(f"Loading cached Markdown for {file}...")
                        with open(md_filepath, "r", encoding="utf-8") as f:
                            clean_md_content = f.read()
                    else:
                        raw_md_content = process_pdf_to_markdown(file_path)
                        clean_md_content = clean_markdown_text(raw_md_content)
                        
                        with open(md_filepath, "w", encoding="utf-8") as f:
                            f.write(clean_md_content)
                        print(f"Saved cleaned Markdown to {md_filepath}")
                    
                    md_header_splits = markdown_splitter.split_text(clean_md_content)
                    splits = text_splitter.split_documents(md_header_splits)
                    
                    for split in splits:
                        split.metadata["source"] = file
                        split.metadata["language"] = language
                        
                    all_chunks.extend(splits)
                except Exception as e:
                    print(f"Failed to process {file}: {e}")

    if not all_chunks:
        print("No chunks created. Ensure there are valid PDFs in the documents directories.")
        return

    print(f"Embedding {len(all_chunks)} chunks using {EMBED_MODEL_NAME}...")
    
    # Választás a modelltől függően:
    if "bge-m3" in EMBED_MODEL_NAME or "nomic" in EMBED_MODEL_NAME:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=absolute_db_path
    )
    
    print(f"Successfully ingested {len(all_chunks)} chunks into {absolute_db_path}")

if __name__ == "__main__":
    data_directory = os.path.join(PROJECT_ROOT, "documents")
    os.makedirs(os.path.join(data_directory, "en"), exist_ok=True)
    os.makedirs(os.path.join(data_directory, "hu"), exist_ok=True)
    
    ingest_documents(data_directory)