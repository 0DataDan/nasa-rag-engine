import os
import sys
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# 1. Target PDF Configuration
if len(sys.argv) > 1:
    pdf_filename = sys.argv[1]
else:
    # Replace with your filename if not passing it via terminal argument
    pdf_filename = "NASA_ARSET_Course_Notes3.pdf"

if not os.path.exists(pdf_filename):
    print(f"Error: File '{pdf_filename}' not found in current directory.")
    sys.exit(1)

print(f"Processing '{pdf_filename}'...")

# 2. Extract Text Page-by-Page
reader = PdfReader(pdf_filename)
pages_data = []

for page_idx, page in enumerate(reader.pages):
    text = page.extract_text()
    if text and text.strip():
        pages_data.append((page_idx + 1, text.strip()))

print(f"Extracted text from {len(pages_data)} pages.")

# 3. Chunking Strategy (Sliding Window ~500 chars with 100 char overlap)
chunks = []
metadatas = []

chunk_size = 500
chunk_overlap = 100

for page_num, page_text in pages_data:
    start = 0
    while start < len(page_text):
        end = start + chunk_size
        chunk = page_text[start:end].strip()
        
        if len(chunk) > 40:  # Ignore tiny artifact fragments
            chunks.append(chunk)
            metadatas.append({
                "source": pdf_filename,
                "page": page_num
            })
        
        start += (chunk_size - chunk_overlap)

print(f"Generated {len(chunks)} text chunks.")

# 4. Connect to ChromaDB and Append
client = chromadb.PersistentClient(path="chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_collection(
    name="nasa_remote_sensing",
    embedding_function=embedding_fn
)

current_count = collection.count()
print(f"Existing chunks in collection: {current_count}")

# Generate unique IDs continuing from existing count
ids = [f"doc_{pdf_filename}_p{m['page']}_c{i}" for i, m in enumerate(metadatas)]

# Add to vector store in batches of 100
batch_size = 100
for i in range(0, len(chunks), batch_size):
    collection.add(
        documents=chunks[i:i + batch_size],
        metadatas=metadatas[i:i + batch_size],
        ids=ids[i:i + batch_size]
    )

new_total = collection.count()
print(f"Successfully added {len(chunks)} chunks.")
print(f"Updated total chunks in 'nasa_remote_sensing': {new_total}")