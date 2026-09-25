import chromadb
from sentence_transformers import SentenceTransformer

# 1. Inisialisasi Model Embedding yang sama dengan saat ingest
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Hubungkan ke Vector Store yang sudah ada
client = chromadb.PersistentClient(path="./vector_store")

def search(query_text, collection_name):
    print(f"\nSearching in {collection_name} for: '{query_text}'")
    
    # Ubah pertanyaan user menjadi vektor
    query_vector = model.encode([query_text]).tolist()
    
    # Ambil koleksi yang sesuai
    collection = client.get_collection(name=collection_name)
    
    # Cari 2 hasil yang paling mirip (top 2)
    results = collection.query(
        query_embeddings=query_vector,
        n_results=2
    )
    
    # Tampilkan hasil
    for i in range(len(results['documents'][0])):
        print(f"Result {i+1}: {results['documents'][0][i]}")
        print(f"Metadata: {results['metadatas'][0][i]}")
        print("-" * 20)

if __name__ == "__main__":
    # Mari kita tes dengan tiga pertanyaan berbeda untuk tiga koleksi
    
    # Tes 1: Cari bukti siswa
    search("Who is struggling with inverse functions?", "student_evidence")
    
    # Tes 2: Cari pengetahuan PCK
    search("What is a common misconception in composite functions?", "pck_knowledge")
    
    # Tes 3: Cari dukungan MTSS
    search("How to provide support for Tier 2?", "mtss_support")
