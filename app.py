import streamlit as st
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------ UI SETUP ------------------
st.set_page_config(page_title="Akshasutra", layout="wide")

st.title("🧠 Akshasutra — Mechanical Engineering AI Assistant")
st.caption("Ask anything from SOM, Thermodynamics, FEM, Robotics & more")

# ------------------ API KEY ------------------
api_key = st.text_input("🔑 Enter your Gemini API Key:", type="password")

if not api_key:
    st.warning("Please enter your Gemini API key to continue")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# ------------------ DOWNLOAD VECTOR DB ------------------
if not os.path.exists("mech_vector_db"):
    st.info("📥 Downloading knowledge base (first time setup)... Please wait ⏳")
    
    import gdown
    
    folder_url = "https://drive.google.com/drive/folders/1Yd55ag9r83Fkmf55Kmro-yK_o06dNfqn?usp=sharing"
    
    gdown.download_folder(folder_url, quiet=False)

# ------------------ LOAD EMBEDDING ------------------
@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding = load_embedding()

# ------------------ LOAD VECTOR DB ------------------
@st.cache_resource
def load_vectorstore():
    return FAISS.load_local(
        "mech_vector_db",
        embedding,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ------------------ LLM ------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

# ------------------ RAG CHAIN ------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ------------------ CHAT MEMORY ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------ INPUT ------------------
query = st.chat_input("💬 Ask your Mechanical Engineering question...")

if query:
    with st.spinner("Thinking..."):
        result = qa({"query": query})
        
        answer = result["result"]
        sources = result["source_documents"]

        st.session_state.chat_history.append((query, answer, sources))

# ------------------ DISPLAY ------------------
for q, a, srcs in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    
    with st.chat_message("assistant"):
        st.write(a)

        st.markdown("### 📚 Sources")
        for doc in srcs[:3]:
            st.write(
                f"• {doc.metadata.get('book_name', 'Unknown')} "
                f"(Page {doc.metadata.get('page', '-')})"
            )