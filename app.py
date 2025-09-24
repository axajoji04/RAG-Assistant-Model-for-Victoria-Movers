import streamlit as st
from langchain_community.llms import Ollama
import requests

st.set_page_config(page_title=" RAG Assistant Model ", page_icon="🚀")

st.title(" 🏠 RAG Assistant Model for Victorias Movers ")
st.markdown("**No API keys • Local Ollama • Simple Version**")

# Check Ollama
def check_ollama():
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return True, [model['name'] for model in models]
    except:
        pass
    return False, []

# Sidebar
with st.sidebar:
    st.header("Ollama Status")
    ollama_running, models = check_ollama()
    
    if ollama_running and models:
        st.success(f"✅ Ollama running with {len(models)} models")
        selected_model = st.selectbox("Select Model:", models)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
        
        if st.button("Reset Chat"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.error("❌ Ollama not running")
        st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Company info
company_info = """
Victoria On Move is a professional moving and relocation service company.
They provide:
- Residential moving services
- Commercial relocations
- Packing and unpacking services
- Loading and transportation
- Storage solutions
- Local and long-distance moves
Contact them for reliable moving solutions in the Victoria area.
"""

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Victoria On Move..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            llm = Ollama(model=selected_model, temperature=temperature)
            
            full_prompt = f"""
            You are a helpful assistant for Victoria On Move moving company.
            
            Company Information:
            {company_info}
            
            User Question: {prompt}
            
            Please provide a helpful, professional answer based on the company information above.
            Keep your response concise and relevant to moving services.
            """
            
            response = llm.invoke(full_prompt)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
