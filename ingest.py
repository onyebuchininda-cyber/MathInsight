import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os

# 1. Inisialisasi Model Embedding
# Model ini bertugas mengubah teks menjadi angka (vektor)
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Inisialisasi ChromaDB
# Data akan disimpan di folder vector_store
client = chromadb.PersistentClient(path="./vector_store")

def ingest_csv(file_path, collection_name):
    print(f"Ingesting {file_path} into collection {collection_name}...")
    
    # Baca CSV
    df = pd.read_csv(file_path)
    
    # Kita buat satu kolom 'text' yang menggabungkan informasi penting agar mudah dicari
    # Ini disebut sebagai 'document' yang akan di-embed
    if collection_name == "student_evidence":
        df['text'] = df.apply(lambda x: f"Student {x['student_id']} ({x['grade']}) - Skill: {x['skill']}. Observation: {x['observation']}. Progress: {x['progress']}", axis=1)
    elif collection_name == "pck_knowledge":
        df['text'] = df.apply(lambda x: f"Skill: {x['skill']}. Typical Error: {x['typical_error']}. Misconception: {x['possible_misconception']}. Prompt: {x['diagnostic_prompt']}", axis=1)
    elif collection_name == "mtss_support":
        df['text'] = df.apply(lambda x: f"Support Type: {x['support_type']} ({x['tier_context']}). Description: {x['description']}", axis=1)
    
    # Buat koleksi di ChromaDB (atau ambil jika sudah ada)
    collection = client.get_or_create_collection(name=collection_name)
    
    # Proses embedding dan simpan ke ChromaDB
    documents = df['text'].tolist()
    ids = [str(i) for i in range(len(documents))]
    embeddings = model.encode(documents).tolist()
    
    # Simpan metadata agar kita bisa melacak source-nya nanti
    metadatas = df.drop(columns=['text']).to_dict(orient='records')
    
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Successfully ingested {len(documents)} records into {collection_name}.")

if __name__ == "__main__":
    # Daftar file yang akan diproses
    files_to_ingest = {
        "data/student_evidence.csv": "student_evidence",
        "data/pck_knowledge.csv": "pck_knowledge",
        "data/mtss_support.csv": "mtss_support"
    }
    
    for path, collection in files_to_ingest.items():
        ingest_csv(path, collection)
    
    print("\nAll data has been vectorized and stored in ./vector_store")
