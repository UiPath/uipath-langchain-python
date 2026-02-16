# scripts/run_epas_agent_with_rag_real.py
import os
from dotenv import load_dotenv
from src.rag.chain import EPASRAGChain
try:
    from uipath import UiPath
except ImportError:
    raise RuntimeError("❌ UiPath SDK non installato. Installa `uipath` per eseguire il workflow reale.")

# =====================================================================
# CONFIGURAZIONE
# =====================================================================
load_dotenv()
UIPATH_URL = os.getenv("UIPATH_URL")
CLIENT_ID = os.getenv("UIPATH_CLIENT_ID")
CLIENT_SECRET = os.getenv("UIPATH_CLIENT_SECRET")
PROCESS_NAME = os.getenv("UIPATH_RPA_WORKFLOW", "RPA.Workflow")
FOLDER_PATH = os.getenv("UIPATH_FOLDER_PATH", "Shared")

# Singola domanda da testare
QUESTION = "What are maintenance organization approval requirements?"

print("UIPATH_URL:", UIPATH_URL)
print("CLIENT_ID:", CLIENT_ID)
print("CLIENT_SECRET:", CLIENT_SECRET)

# =====================================================================
# INIZIALIZZAZIONE RAG LOCALE
# =====================================================================
print("\n🔹 Inizializzazione EPAS RAG Chain con vector store locale...")
rag = EPASRAGChain("data/vectorstore")
print("✅ Vector store caricato con successo.\n")

# =====================================================================
# AUTENTICAZIONE
# =====================================================================
try:
    print("🔑 Autenticazione su UiPath Cloud...")
    sdk = UiPath(
        base_url=UIPATH_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scope="OR.Jobs OR.Execution",
        debug=True
    )
    print("✅ Autenticazione riuscita")
except Exception as e:
    print(f"❌ Errore autenticazione: {e}")
    exit(1)

# =====================================================================
# ESECUZIONE WORKFLOW
# =====================================================================
print(f"\n🔹 Invio query all'EPAS Agent: {QUESTION}")

# Step 1 — Recupero contesto dal RAG
rag_result = rag.query(QUESTION)
input_arguments = {
    "in_query": QUESTION,
    "in_context": rag_result["answer"]
}

print("\n📋 Contesto RAG:")
print("=" * 60)
print(rag_result["answer"])
print("=" * 60)

# Step 2 — Avvio workflow
try:
    print(f"\n🚀 Avvio workflow '{PROCESS_NAME}'...")
    job = sdk.processes.invoke(
        PROCESS_NAME,
        input_arguments=input_arguments,
        folder_path=FOLDER_PATH
    )
    job_id = job.id
    print(f"\n✅ Job avviato con successo!")
    print(f"📄 Job ID: {job_id}")
    print(f"📊 Stato iniziale: {job.state}")
    print(f"\n💡 Puoi monitorare il job su UiPath Orchestrator con l'ID: {job_id}")
except Exception as e:
    print(f"\n❌ Errore avvio workflow: {e}")
    exit(1)

print("\n✅ Script terminato - Job avviato e in esecuzione")