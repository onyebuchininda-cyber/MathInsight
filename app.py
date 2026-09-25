import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv
import pandas as pd

# Load API Key dari file .env
load_dotenv()

# Pengaturan Halaman Streamlit
st.set_page_config(page_title="MathInsight", page_icon="📐")

# Pemetaan nama teknis ke bahasa ramah guru
SOURCE_MAP = {
    "student_evidence": "Bukti Belajar Siswa",
    "pck_knowledge": "Pengetahuan Pedagogis (PCK)",
    "mtss_support": "Kerangka Dukungan (MTSS)"
}

# 1. Fungsi untuk Ingest Data (Membangun Vector Store)
def ingest_data(model, client):
    """Fungsi ini membangun kembali database jika belum ada di server."""
    files_to_ingest = {
        "data/student_evidence.csv": "student_evidence",
        "data/pck_knowledge.csv": "pck_knowledge",
        "data/mtss_support.csv": "mtss_support"
    }
    
    for path, col_name in files_to_ingest.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            # Buat teks dokumen untuk embedding
            if col_name == "student_evidence":
                df['text'] = df.apply(lambda x: f"Student {x['student_id']} ({x['grade']}) - Skill: {x['skill']}. Observation: {x['observation']}. Progress: {x['progress']}", axis=1)
            elif col_name == "pck_knowledge":
                df['text'] = df.apply(lambda x: f"Skill: {x['skill']}. Typical Error: {x['typical_error']}. Misconception: {x['possible_misconception']}. Prompt: {x['diagnostic_prompt']}", axis=1)
            elif col_name == "mtss_support":
                df['text'] = df.apply(lambda x: f"Support Type: {x['support_type']} ({x['tier_context']}). Description: {x['description']}", axis=1)
            
            collection = client.get_or_create_collection(name=col_name)
            
            documents = df['text'].tolist()
            ids = [str(i) for i in range(len(documents))]
            embeddings = model.encode(documents).tolist()
            metadatas = df.drop(columns=['text']).to_dict(orient='records')
            
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

# 2. Inisialisasi Resource (Caching agar efisien)
@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./vector_store")
    
    # CEK: Jika koleksi utama tidak ada, jalankan ingest_data secara otomatis
    try:
        client.get_collection(name="student_evidence")
    except:
        # Jika error (berarti database kosong), bangun databasenya sekarang
        ingest_data(model, client)
        
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return model, client, groq_client

model, client, groq_client = load_resources()

def clean_metadata(meta):
    if not meta: return ""
    lines = [f"{k.replace('_', ' ').title()}: {v}" for k, v in meta.items()]
    return "\n".join(lines)

# Tampilan UI
st.title("📐 MathInsight")
st.markdown("""
Selamat datang di **MathInsight**. Chatbot ini membantu guru menganalisis bukti belajar siswa 
dengan menghubungkannya ke PCK dan konteks MTSS secara objektif.
""")

query = st.text_input("Masukkan pertanyaan Anda tentang siswa atau materi:", placeholder="Contoh: Apa saran untuk membantu siswa S02 yang kesulitan komposisi fungsi?")

if query:
    with st.spinner("MathInsight sedang berpikir..."):
        collections = ["student_evidence", "pck_knowledge", "mtss_support"]
        query_vector = model.encode([query]).tolist()
        
        retrieved_data = []
        full_context_for_ai = ""

        for col_name in collections:
            try:
                collection = client.get_collection(name=col_name)
                results = collection.query(query_embeddings=query_vector, n_results=1)
                
                if results['documents'][0]:
                    doc = results['documents'][0][0]
                    meta = results['metadatas'][0][0]
                    retrieved_data.append({
                        "Sumber": SOURCE_MAP.get(col_name, col_name),
                        "Informasi": doc,
                        "Detail": clean_metadata(meta)
                    })
                    full_context_for_ai += f"\nSource [{col_name}]: {doc}\nMetadata: {meta}\n"
            except Exception as e:
                st.error(f"Error mencari di {col_name}: {e}")

        if retrieved_data:
            system_prompt = (
                "Anda adalah asisten ahli pedagogi matematika untuk guru. "
                "Tugas Anda adalah membantu guru menganalisis data siswa menggunakan bukti yang diberikan. "
                "\n\nPRINSIP KETAT:\n"
                "1. JANGAN mendiagnosis siswa secara medis atau psikologis. Gunakan bahasa observasi.\n"
                "2. Anggap 'Possible Misconception' sebagai 'Hipotesis Pedagogis', bukan kepastian.\n"
                "3. JANGAN menentukan Tier MTSS secara otomatis. Berikan opsi dukungan yang relevan dari data.\n"
                "4. Jawaban HARUS berdasarkan bukti (retrieved evidence) yang disediakan. Jika bukti tidak cukup, nyatakan 'insufficient evidence'.\n"
                "5. PENTING: Ubah nama sumber teknis menjadi bahasa yang ramah guru:\n"
                "   - [student_evidence] menjadi 'Bukti Belajar Siswa'\n"
                "   - [pck_knowledge] menjadi 'Pengetahuan Pedagogis (PCK)'\n"
                "   - [mtss_support] menjadi 'Kerangka Dukungan (MTSS)'\n"
                "6. Sajikan informasi dalam format tabel atau daftar yang rapi dan profesional."
            )
            
            user_prompt = f"Pertanyaan Guru: {query}\n\nBukti yang ditemukan:\n{full_context_for_ai}\n\nBerikan analisis dan saran dukungan yang tepat untuk guru."

            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model="openai/gpt-oss-120b",
                )
                ai_response = chat_completion.choices[0].message.content
                st.subheader("🤖 Analisis MathInsight")
                st.write(ai_// la respons)
                st.write(ai_response)
                
                with st.expander("Lihat Bukti Data yang Digunakan"):
                    df_evidence = pd.DataFrame(retrieved_data)
                    st.table(df_evidence)
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menghubungi AI: {e}")
        else:
            st.info("Maaf, tidak ditemukan bukti atau pengetahuan yang cukup untuk menjawab pertanyaan ini.")
