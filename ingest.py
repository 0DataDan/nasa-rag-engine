import os
import re
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

def sanitize_media_references(text: str) -> str:
    """
    Strips out references to missing figures, tables, or video embeds
    that provide no text value to the retrieval model.
    """
    text = re.sub(r'[\(\[]?\b(?:see\vert{}refer to\vert{}shown in)?\s*(?:figure\vert{}fig\.\vert{}table\vert{}video\vert{}diagram\vert{}image)\s*\d+[\.\d*]*[\)\]]?', '', text, flags=re.IGNORECASE)
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
        print(f"[!] Directory '{data_dir}        print(f"[!] Directory '{data_dir}        print(f"[!RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".         separators=["\n\n", "\n", ".           )
    
    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    l_ch    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c    all_c  ):
                                        , filename)
        doc = fitz.open(pdf_path)
        print(f"--> Reading '{filenam        print(f"--> R)...        print(f"--> Reading '{filenam        print(f"--> R)...        print(f"--> Reaoc)):
                                                                     ("t                                                    rences(raw_tex                               len(cl                                                                     ("t                                                    r            
            for idx, spli            for idx, spli      ):
                file_chunk_count += 1
                chunk_record = {
                    "ch                    "ch                    "ch                    "ch                    "ch                    "ch                    "ch                    "ch                    "ch                    "ch                    "ch                    "ch                    "ch                    "ch                t)
                    }
                }
                                            rd)                                            rd)                                            rd)                                            rd)        == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "data")
    chunks = process_pdf_folder(data_folder, chunk_size=600, chunk_overlap=120)
    print(f"\n[+] Total chunks created across all documents: {len(chunks)}")
    if chunks:
        print("\n--- SAMPLE CHUNK PREVIEW ---")
        print(f"Chunk ID : {        print(f"Chunk ID : {     print        print(f"Chunk ID 'metadata']['source']} (Page {chunks[0]['metadata']['page']})")
        print(f"Length   : {chunks[0]['metadata']['char_length']} characters")
        print("Text Content:")
        print(f"\"{chunks[0]['text']}\"")
        print("----------------------------\n")
