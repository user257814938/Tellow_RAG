from typing import List, Optional, Iterator
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import LanceDB
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.document_loaders import BaseLoader
import lancedb
from docling.document_converter import DocumentConverter

from config import Config

# ==============================================================================
# SECTION 1: CUSTOM DOCUMENT LOADER
# ==============================================================================
class SimpleDoclingLoader:
    """
    Custom loader using Docling SDK directly to avoid dependency hell.
    This loader handles various file formats including PDF, Docx, and Audio.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[Document]:
        print(f"Converting {self.file_path} with Docling...")
        result = self._converter.convert(self.file_path)
        # Use export_to_markdown() as it provides good structure for LLMs
        md_content = result.document.export_to_markdown()
        yield Document(
            page_content=md_content,
            metadata={"source": self.file_path}
        )

    def load(self) -> List[Document]:
        return list(self.lazy_load())

# ==============================================================================
# SECTION 2: RAG SYSTEM CORE
# ==============================================================================
class RAGSystem:
    def __init__(self):
        """
        Initialize the RAG System components:
        - Embeddings: OpenAI
        - LLM: ChatOpenAI
        - Database Connection: LanceDB
        """
        self.embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        self.llm = ChatOpenAI(model=Config.LLM_MODEL)
        self.vector_store = None
        self._db = lancedb.connect(Config.LANCEDB_URI)

    # --------------------------------------------------------------------------
    # Phase A: Document Loading & Splitting
    # --------------------------------------------------------------------------
    def load_documents(self, source: str) -> List[Document]:
        """Loads documents using Docling from a URL or file path."""
        print(f"Loading documents from: {source}...")
        # Use our custom loader
        loader = SimpleDoclingLoader(file_path=source)
        docs = loader.load()
        
        # Split documents for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(docs)
        print(f"Loaded and split into {len(splits)} chunks.")
        return splits

    # --------------------------------------------------------------------------
    # Phase B: Vector Store Indexing
    # --------------------------------------------------------------------------
    def setup_vector_store(self, documents: List[Document]):
        """Initializes or updates the LanceDB vector store."""
        print("Indexing documents into LanceDB...")
        # Create table if it doesn't exist, or open it
        self.vector_store = LanceDB.from_documents(
            documents,
            self.embeddings,
            connection=self._db,
            table_name=Config.TABLE_NAME
        )
        print("Indexing complete.")

    # --------------------------------------------------------------------------
    # Phase C: Retrieval Chain Construction
    # --------------------------------------------------------------------------
    def get_rag_chain(self):
        """Creates the retrieval chain."""
        if not self.vector_store:
            # Try to load existing table if possible
            try:
                self.vector_store = LanceDB(
                    connection=self._db,
                    embedding=self.embeddings,
                    table_name=Config.TABLE_NAME
                )
            except Exception:
                pass
            
            if not self.vector_store: 
                raise ValueError("Vector store not initialized. Load documents first.")

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": Config.SEARCH_K}
        )

        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}
""")

        document_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        return retrieval_chain

    # --------------------------------------------------------------------------
    # Phase D: Query Execution
    # --------------------------------------------------------------------------
    def query(self, question: str) -> dict:
        """
        Queries the RAG system.
        Returns a dictionary containing:
        - key 'answer': The generated answer string.
        - key 'context': The list of retrieved documents.
        - key 'scores': List of L2 distance scores for retrieved docs.
        - key 'top_k': The K value used for retrieval.
        """
        if not self.vector_store:
            # Try to load existing table if possible
            try:
                self.vector_store = LanceDB(
                    connection=self._db,
                    embedding=self.embeddings,
                    table_name=Config.TABLE_NAME
                )
            except Exception:
                pass
            
            if not self.vector_store: 
                raise ValueError("Vector store not initialized. Load documents first.")

        # 1. Retrieve with Scores (L2 Distance)
        # LanceDB default is L2. Lower is better.
        k_value = Config.SEARCH_K
        docs_and_scores = self.vector_store.similarity_search_with_score(
            question, 
            k=k_value
        )
        
        # Unpack results
        docs = [doc for doc, score in docs_and_scores]
        scores = [score for doc, score in docs_and_scores]

        # 2. Prepare Generation
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}
""")
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # 3. Generate Answer
        response = document_chain.invoke({
            "input": question,
            "context": docs
        })

        # 4. Return enriched response
        return {
            "answer": response,
            "context": docs,
            "scores": scores,
            "top_k": k_value
        }
