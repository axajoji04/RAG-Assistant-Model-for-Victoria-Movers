import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import requests
from bs4 import BeautifulSoup
import time
import sys

print("🚀 100% FREE RAG SYSTEM - No API keys needed!")
print("📋 Requirements:")
print("   1. pip install sentence-transformers langchain-huggingface langchain-community")
print("   2. Install Ollama: https://ollama.ai/download")
print("   3. Pull a model: ollama pull llama3.2")
print()

# URLs to load
urls = [
    "https://www.manwithavan.com.au/",
    "https://www.manwithavan.com.au/about"
]

print("📥 Loading documents from URLs...")

def load_documents_safely(urls):
    documents = []
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    for url in urls:
        try:
            print(f"🌐 Loading: {url}")
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                for script in soup(["script", "style", "nav", "header", "footer"]):
                    script.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_text = ' '.join(chunk for chunk in chunks if chunk)
                if len(clean_text.strip()) > 200:
                    doc = Document(
                        page_content=clean_text,
                        metadata={"source": url}
                    )
                    documents.append(doc)
                    print(f"✅ Success: {len(clean_text)} characters")
                else:
                    print(f"⚠️ Too little content from {url}")
            else:
                print(f"❌ HTTP {response.status_code} for {url}")
        except Exception as e:
            print(f"❌ Error loading {url}: {e}")
            continue
        time.sleep(1)
    return documents

try:
    data = load_documents_safely(urls)
    if not data:
        print("❌ No documents loaded")
        sys.exit(1)
    print(f"✅ Loaded {len(data)} documents successfully")
    for i, doc in enumerate(data):
        print(f"Document {i+1}: {len(doc.page_content)} characters from {doc.metadata['source']}")
except Exception as e:
    print(f"❌ Error loading documents: {e}")
    sys.exit(1)

print("\n📝 Splitting documents...")
try:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs = text_splitter.split_documents(data)
    print(f"✅ Created {len(docs)} text chunks")
    if docs:
        print(f"Sample chunk: {docs[0].page_content[:200]}...")
except Exception as e:
    print(f"❌ Error splitting documents: {e}")
    sys.exit(1)

print("\n🔍 Creating vector embeddings (100% local and free)...")
try:
    # Free local embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("📦 Using local embedding model")
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=None
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    print("✅ Vector store created successfully!")
except Exception as e:
    print(f"❌ Error creating vectorstore: {e}")
    sys.exit(1)

print("\n🤖 Setting up QA chain with Ollama...")
try:
    system_prompt = (
        "You are an assistant for question answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Using Ollama (100% free local LLM)
    llm = Ollama(
        model="llama3.2",  # or "llama3.1", "mistral", etc.
        temperature=0.3
    )
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    print("✅ QA chain created successfully with Ollama!")
except Exception as e:
    print(f"❌ Error creating QA chain: {e}")
    print("ℹ️ Make sure Ollama is running and you have pulled a model:")
    print("   ollama pull llama3.2")
    sys.exit(1)

# Test the system
test_queries = [
    "What kind of services do they provide?",
    "How can I contact them?",
    "What is this company about?",
    "Do they provide moving services?"
]

print("\n" + "="*50)
print("🚀 TESTING 100% FREE RAG SYSTEM")
print("="*50)

for query in test_queries:
    print(f"\n❓ Query: {query}")
    print("-" * 30)
    try:
        response = rag_chain.invoke({"input": query})
        print(f"📝 Answer: {response['answer']}")
    except Exception as e:
        print(f"❌ Error answering query: {e}")

print("\n🎉 100% FREE RAG system working!")
print("💡 This setup uses:")
print("   • Local embeddings (Hugging Face)")
print("   • Local LLM (Ollama)")
print("   • No API keys or quotas!")
print("   • Runs entirely on your machine!")
