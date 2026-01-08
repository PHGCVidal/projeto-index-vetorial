# buscar_trechos.py

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# --- CONFIGURAÇÕES ---
VECTORDB_PATH = "base_vetorial_local"
MODEL_NAME_EMBEDDING = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# --- FIM DAS CONFIGURAÇÕES ---

# 1. Carregar o modelo de embedding
print("Carregando modelo de embedding...")
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME_EMBEDDING)

# 2. Carregar a base de dados vetorial que já existe
print("Carregando base de dados vetorial local...")
vectordb = Chroma(persist_directory=VECTORDB_PATH, embedding_function=embeddings)

# 3. Criar o "buscador" (retriever)
# k=4 significa que vamos buscar os 4 trechos mais relevantes
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# --- LOOP DE PERGUNTAS ---
print("-" * 50)
print("Buscador de Trechos pronto. Digite 'sair' para terminar.")
while True:
    sua_pergunta = input("\nDigite a pergunta para buscar nos documentos: ")
    if sua_pergunta.lower() == 'sair':
        break

    print("\nBuscando os trechos mais relevantes...")
    
    # 4. Executar a busca e obter os documentos (trechos)
    # Esta é a parte que funciona no seu PC!
    documentos_relevantes = retriever.get_relevant_documents(sua_pergunta)
    
    # 5. Imprimir os resultados para você copiar e colar
    print("-" * 50)
    print("TRECHOS ENCONTRADOS (Contexto para a IA):")
    print("-" * 50)
    
    for i, doc in enumerate(documentos_relevantes):
        print(f"--- TRECHO {i+1} (Fonte: {doc.metadata.get('source', 'N/A')}, Página: {doc.metadata.get('page', 'N/A')}) ---\n")
        print(doc.page_content)
        print("\n" + "="*50 + "\n")
        
    print("\n\nCrie seu prompt para o ChatGPT usando a pergunta e os trechos acima.")