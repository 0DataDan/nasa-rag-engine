import os
import re
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

def sanitize_media_references(text: str) -> str:
    """
    Strips out references to missing figures, tables, or video embeds
    that provide no text value to the retrieval model.
    """
    text = re.sub(r'[\(\[]?\b(?:see|refer to|shown in)?\s*(?:figure|fig\.|table|video|diagram|image)\s*\d+[\.\d*]*[\)\]]?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Figure|Fig\.|Table|Video|Diagram|Image)\s*\d+[\.\d*]*\s*[:-]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_pdf_folder(data_dir: str, chunk_size: int = 600, chunk_overlap: int = 120):
    """
    Iterates through all PDFs in the data folder, cleans the text,
    and slices them into structured chunks with metadata.
    """
    if not os.path.exists(data_dir):
        print(f"[!] Directory '{data_dir}' does not exist.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " "],
        length_function=len
    )
    
    all_chunks = []
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"[!] No PDF files found in '{data_dir}'.")
        return all_chunks

    print(f"[*] Found {len(pdf_files)} PDF file(s) to process.\n")

    for filename in sorted(pdf_files):
        pdf_path = os.path.join(data_dir, filename)
        doc = fitz.open(pdf_path)
        print(f"--> Reading '{filename}' ({len(doc)} pages)...")
        
        file_chunk_count = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            raw_text = page.get_text("text")
            cleaned_text = sanitize_media_references(raw_text)
            
            if len(cleaned_text) < 40:
                continue
                
            page_splits = text_splitter.split_text(cleaned_text)
            
            for idx, split_text in enumerate(page_splits):
                file_chunk_count += 1
                chunk_record = {
                    "chunk_id": f"{filename}_p{page_num + 1}_c{idx + 1}",
                    "text": split_text,
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1,
                        "chunk_index": idx + 1,
                        "char_length": len(split_text)
                    }
                }
                all_chunks.append(chunk_record)
                
        doc.close()
        print(f"    Generated {file_chunk_count} chunks from {filename}.")

    return all_chunks

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "data")
    chunks = process_pdf_folder(data_folder, chunk_size=600, chunk_overlap=120)
    print(f"\n[+] Total chunks created across all documents: {len(chunks)}")
    if chunks:
        print("\n--- SAMPLE CHUNK PREVIEW ---")
        print(f"Chunk ID : {chunks[0]['chunk_id']}")
        print(f"Source   : {chunks[0]['metadata']['source']} (Page {chunks[0]['metadata']['page']})")
        print(f"Length   : {chunks[0]['metadata']['char_length']} characters")
        print("Text Content:")
        print(f"\"{chunks[0]['text']}\"")
        print("----------------------------\n")