import os
import sys
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# 1. Target PDF Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) > 1:
    pdf_input = sys.argv[1]
else:
    # Default fallback pointing inside data/
    pdf_input = os.path.join(current_dir, "data", "NASA_ARSET_Course_Notes3.pdf")

# Resolve absolute path safely
pdf_path = os.path.abspath(pdf_input) if os.path.isabs(pdf_input) else os.path.join(os.getcwd(), pdf_input)

if not os.path.exists(pdf_path):
    # Secondary fallback check inside data/ folder
    fallback_path = os.path.join(current_dir, "data", os.path.basename(pdf_input))
    if os.path.exists(fallback_path):
        pdf_path = fallback_path
    else:
        print(f"[!] Error: File '{pdf_input}' not found.")
        sys.exit(1)

filename = os.path.basename(pdf_path)
print(f"[*] Processing '{filename}'...")

# 2. Extract Text Page-by-Page
reader = PdfReader(pdf_path)
pages_data = []

for page_idx, page in enumerate(reader.pages):
    text = page.extract_text()
    if text and text.strip():
        pages_data.append((page_idx + 1, text.strip()))

print(f"[*] Extracted text from {len(pages_data)} pages.")

# 3. Chunking Strategy (Sliding Window ~500 chars with 100 char overlap)
chunks = []
metadatas = []
ids = []

chunk_size = 500
chunk_overlap = 100

for page_num, page_text in pages_data:
    start = 0
    chunk_idx = 1
    while start < len(page_text):
        end = start + chunk_size
        chunk = page_text[start:end].strip()
        
        if len(chunk) > 40:  # Ignore tiny fragments
            chunks.append(chunk)
            metadatas.append({
                "source": filename,
                "page": page_num,
                "chunk_index": chunk_idx,
                "char_length": len(chunk)
            })
            ids.append(f"{filename}_p{page_num}_c{chunk_idx}")
            chunk_idx += 1
        
        start += (chunk_size - chunk_overlap)

if not chunks:
    print(f"[!] No valid text chunks could be generated from '{filename}'.")
    sys.exit(0)

print(f"[*] Generated {len(chunks)} text chunks.")

# 4. Connect to ChromaDB and Upsert
db_folder = os.path.join(current_dir, "chroma_db")
client = chromadb.PersistentClient(path=db_folder)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_or_create_collection(
    name="nasa_remote_sensing",
    embedding_function=embedding_fn
)

current_count = collection.count()
print(f"[*] Pre-existing chunks in collection: {current_count}")

# Upsert in batches of 100 to handle large documents cleanly
batch_size = 100
for i in range(0, len(chunks), batch_size):
    collection.upsert(
        documents=chunks[i:i + batch_size],
        metadatas=metadatas[i:i + batch_size],
        ids=ids[i:i + batch_size]
    )

new_total = collection.count()
print(f"[+] Successfully indexed {len(chunks)} chunks from '{filename}'.")
print(f"[+] Updated total chunks in 'nasa_remote_sensing': {new_total}\n")