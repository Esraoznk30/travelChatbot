# main.py
import os
from typing import List, Dict, Tuple
import gradio as gr
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from dotenv import load_dotenv
    load_dotenv()
    print(".env dosyası yüklendi.")
except ImportError:
    print("UYARI: 'python-dotenv' kütüphanesi yüklü değil. API anahtarı hala os.environ'da aranıyor.")

# Kodu daha temiz hale getirmek için API anahtarı atamasını kaldırıyoruz.
# LangChain, anahtarı os.environ'dan otomatik olarak alacaktır.
# ----------------------------
# GEMINI API KEY KONTROLÜ
# ----------------------------
if not os.environ.get("GOOGLE_API_KEY"):
    print("KRİTİK UYARI: GOOGLE_API_KEY bulunamadı. LLM çalışmayabilir!")
# ----------------------------
llm = None
USE_LLM = False
try:
    # Güncel LangChain import yapısı
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    USE_LLM = True
except ImportError:
    try:
        from langchain.chat_models import init_chat_model

        llm = init_chat_model("google_genai:gemini-2.0-flash", temperature=0.2)
        USE_LLM = True
    except Exception as e:
        print(f"Gemini LLM yüklenemedi. Hata: {e}")
        llm = None
        USE_LLM = False

# ----------------------------
# Embedding modeli ve ChromaDB (Değişmedi)
# ----------------------------
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

PERSIST_DIR = "./chroma_db"
client = chromadb.PersistentClient(path=PERSIST_DIR)

collection = client.get_or_create_collection(
    name="turkish_rag_v2",
    metadata={"description": "Türkçe seyahat RAG sistemi için embedding"}
)

# ----------------------------
# Örnek seyahat verisi, Chunking, ChromaDB Ekleme (Değişmedi)
# ----------------------------
travel_data = """
İstanbul: Ayasofya, Topkapı Sarayı, Sultanahmet, Galata Kulesi. Boğaz turu ve tarihî yarımada önerilir.
Kapadokya: Göreme, Ürgüp, sıcak hava balon turları gün doğumunda yapılır; peri bacaları ve vadiler gezilmeli.
Antalya: Konyaaltı ve Lara plajları, antik kentler (Perge, Aspendos) ve deniz tatili için ideal.
İzmir: Kordon, Saat Kulesi, Kadifekale; yakın Çeşme ve Alaçatı popüler.
Paris: Eyfel Kulesi, Louvre, Seine kıyısı, romantik yürüyüş rotaları.
Roma: Kolezyum, Vatikan, tarihi merkezde yürüyüş ve mutfak denemeleri.
"""


def simple_split(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
    return chunks


all_chunks = simple_split(travel_data)
doc_metadata = [{"source": f"travel_data_chunk_{i}"} for i in range(len(all_chunks))]


def add_documents_to_db(chunks: List[str], metadata: List[Dict]):
    if collection.count() > 0:
        print("Koleksiyon zaten veri içeriyor, tekrar ekleme yapılmadı.")
        return
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = embedding_model.embed_documents(chunks)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata,
        ids=ids
    )
    print(f"{len(chunks)} adet chunk veritabanına eklendi")


add_documents_to_db(all_chunks, doc_metadata)


# ----------------------------
# Retrieval (Değişmedi)
# ----------------------------
def retrieve_top_k(query: str, k: int = 3) -> List[dict]:
    q_emb = embedding_model.embed_query(query)
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    results = []
    for doc, meta, dist in zip(docs, metas, dists):
        results.append({"doc": doc, "meta": meta, "dist": dist})
    return results


# ----------------------------
# LLM çağrısını ve Konuşma Geçmişi Yönetimi (YENİ)
# ----------------------------

def format_history_for_prompt(history: List[Tuple[str, str]]) -> str:
    """Gradio geçmişini LLM'ye uygun metin formatına dönüştürür."""
    formatted_text = ""
    for user_msg, bot_msg in history:
        # Son kullanıcı sorgusu hariç (o zaten 'question' değişkeninde olacak)
        if user_msg and bot_msg:
            formatted_text += f"Kullanıcı: {user_msg}\nAsistan: {bot_msg}\n"
    return formatted_text


def call_llm_with_context(
        question: str,
        contexts: List[str],
        history: List[Tuple[str, str]]
) -> str:
    """LLM'yi bağlam, soru ve konuşma geçmişi ile çağırır."""

    context_text = "\n\n".join(contexts) if contexts else ""
    history_text = format_history_for_prompt(history[:-1])  # Son sorgu hariç

    # PROMPT GÜNCELLEMESİ: Konuşma geçmişini dahil ediyoruz
    prompt = (
        "Sana bir konuşma geçmişi, bazı harici bilgiler ('Bağlam') ve güncel bir 'Soru' verilecektir. "
        "Soruyu cevaplarken: "
        "1. Öncelikle 'Bağlam'daki bilgileri kullan. "
        "2. Cevabın bağlamda yoksa, konuşma geçmişindeki önceki diyalogları dikkate alarak kendi genel bilgilerinle cevap ver.\n\n"

        "--- KONUŞMA GEÇMİŞİ ---\n"
        f"{history_text}\n"
        "-----------------------\n\n"

        f"Bağlam:\n{context_text}\n\n"
        f"GÜNCEL SORU: {question}\nCevap:"
    )

    if USE_LLM and llm is not None:
        try:
            # LangChain'de bu kadar uzun prompt'ları doğrudan invoke ile göndermek genelde çalışır.
            resp = llm.invoke(prompt)
            return resp.content
        except Exception as e:
            return f"(LLM çağrısında hata: {e})"
    else:
        return contexts[0] if contexts else "LLM kapalı. Bilmiyorum."


# ----------------------------
# Gradio chat fonksiyonu (YENİ)
# ----------------------------
# Gradio'nun ChatInterface'i için fonksiyon imzası (question, history) olmalıdır.
def chat_fn(user_question: str, history: List[Tuple[str, str]]) -> str:
    if not user_question.strip():
        return "Lütfen bir soru yazın."

    # 1. Retrieval (Kaynak Bulma)
    # RAG, mevcut soruya göre yapılır (konuşma geçmişine göre değil, daha karmaşık RAG için o da gerekebilir).
    hits = retrieve_top_k(user_question, k=3)
    contexts = [h["doc"] for h in hits]

    # 2. LLM Çağrısı (Geçmiş dahil)
    answer = call_llm_with_context(user_question, contexts, history)

    # Kaynak bilgisini cevabın sonuna eklemek, kullanıcı için şeffaflık sağlar.
    sources = "\n---\n**Kaynaklar (RAG)**:\n" + "\n".join(
        [f"- {h['doc'][:50]}... (dist={h['dist']:.4f})" for h in hits])

    return answer


# ----------------------------
# Modern Gradio UI (YENİ - ChatInterface)
# ----------------------------
with gr.Blocks(title="Seyahat Chatbotu") as demo:
    gr.Markdown("## Seyahat Chatbotu ")
    if not USE_LLM:
        gr.Markdown(
            "<p style='color: orange;'><strong>UYARI:</strong> Gemini LLM kullanılamıyor. Yalnızca Vektör Araması (RAG) yapılacaktır.</p>"
        )

    # gr.ChatInterface kullanıyoruz, bu önceki konuşmaları otomatik yönetir.
    # fn: Konuşmayı yürütecek fonksiyondur.
    # examples: Başlangıçta göstereceği örnek sorgular.
    # title: Chatbot başlığı.
    gr.ChatInterface(
        fn=chat_fn,
        examples=["İstanbul'da nereler gezilir?", "Kapadokya'da balon turları ne zaman yapılır?", "Paris'te ne yenir?"],
        title="Seyahat Asistanı"
    )

if __name__ == "__main__":
    demo.launch()