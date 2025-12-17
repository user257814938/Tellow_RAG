import streamlit as st
import tempfile
import os
import shutil
import time
from rag_system import RAGSystem

# Page Config
st.set_page_config(page_title="Tellow RAG", layout="wide")

@st.cache_resource
def get_rag_system():
    return RAGSystem()

def save_uploaded_file(uploaded_file):
    """Saves uploaded file to a temporary file and returns the path."""
    try:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(uploaded_file, tmp_file)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def main():
    st.title("🤖 Tellow RAG")
    
    rag = get_rag_system()
    
    # Sidebar: Configuration & Data Loading
    with st.sidebar:
        st.header("Data Source")
        
        # URL Input
        url_input = st.text_input("Document URL", placeholder="https://example.com/doc.pdf")
        
        # File Upload
        uploaded_file = st.file_uploader(
            "Or upload a file", 
            type=[
                'pdf', 'docx', 'pptx', 'xlsx', 'html', 'txt', 'md', 
                'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'wav', 'mp3', 'vtt'
            ]
        )
        
        if st.button("Load Document"):
            source = None
            if uploaded_file:
                with st.spinner("Processing uploaded file..."):
                    temp_path = save_uploaded_file(uploaded_file)
                    if temp_path:
                        source = temp_path
            elif url_input:
                source = url_input
            
            if source:
                try:
                    with st.spinner(f"Loading and indexing {source}..."):
                        documents = rag.load_documents(source)
                        rag.setup_vector_store(documents)
                        st.success("Document indexed successfully! You can now chat.")
                        st.session_state["rag_ready"] = True
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please provide a URL or upload a file.")

    # Main Area: Chat
    st.subheader("Chat with your Document")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display metrics if they exist in the message history
            # Display metrics if they exist in the message history
            if "metrics" in message:
                metrics = message["metrics"]
                with st.expander("📊 Response Metrics"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"⏱️ **Time:** {metrics.get('time', 0):.2f}s")
                    with col2:
                        st.write(f"🔢 **Top K:** {metrics.get('top_k', 'N/A')}")
                        
                    st.write("**📄 Retrieved Documents (Ranked):**")
                    
                    sources = metrics.get("sources", [])
                    scores = metrics.get("scores", [])
                    
                    if not scores:
                        scores = [0.0] * len(sources)
                        
                    for i, (source, score, doc_content) in enumerate(zip(sources, scores, metrics.get("contexts", [""] * len(sources)))):
                        if i >= 1: break # Only show top 1
                        sim_score = max(0, 1 - (score**2 / 2))
                        st.markdown(f"""
                        **Top Result:** 📂 **Source:** `{os.path.basename(source)}`  
                        - 📏 Distance (L2): `{score:.4f}`
                        - 🎯 Similarity: `{sim_score:.1%}`
                        """)
                        with st.expander("📜 View Content Snippet", expanded=False):
                            st.markdown(doc_content)

    # React to user input
    if prompt := st.chat_input("What is this document about?"):
        # Check if RAG is ready
        if not st.session_state.get("rag_ready"):
            st.error("Please load a document first!")
            return

        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    start_time = time.time()
                    
                    # RAG Query
                    result = rag.query(prompt)
                    
                    # Extract answer and metadata
                    answer = result.get("answer", "")
                    context_docs = result.get("context", [])
                    scores = result.get("scores", [])
                    top_k = result.get("top_k", 0)
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    st.markdown(answer)
                    
                    # Display Metrics
                    with st.expander("📊 Response Metrics", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"⏱️ **Time:** {elapsed_time:.2f}s")
                        with col2:
                            st.write(f"🔢 **Top K:** {top_k}")
                            
                        st.write("**📄 Retrieved Document (Top 1):**")
                        
                        # Zip docs and scores. If scores missing (mocking?), handle gracefully
                        if not scores:
                            scores = [0.0] * len(context_docs)
                            
                        for i, (doc, score) in enumerate(zip(context_docs, scores)):
                            if i >= 1: break # Only show top 1
                            source = doc.metadata.get("source", "Unknown")
                            # Calculate similarity % (Approximation for L2 on normalized vectors)
                            # Cosine Sim = 1 - (L2^2 / 2)
                            sim_score = max(0, 1 - (score**2 / 2))
                            
                            st.markdown(f"""
                            **Top Result:** 📂 **Source:** `{os.path.basename(source)}`  
                            - 📏 Distance (L2): `{score:.4f}`
                            - 🎯 Similarity: `{sim_score:.1%}`
                            """)
                            with st.expander("📜 View Content Snippet", expanded=False):
                                st.markdown(doc.page_content)

                    # Add assistant response to chat history with metrics
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "metrics": {
                            "time": elapsed_time,
                            "sources": [doc.metadata.get("source", "Unknown") for doc in context_docs],
                            "scores": scores,
                            "top_k": top_k,
                            "contexts": [doc.page_content for doc in context_docs]
                        }
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
