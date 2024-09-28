####---------------------------------------------RAG Pipeline-----------------------------------------------####

import os
import tempfile
import time
import streamlit as st
import logging
import gc
import base64
import gc
import tempfile
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Optional, List, Dict, Any
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv;load_dotenv()
from langchain_groq import ChatGroq
from langchain.schema import StrOutputParser
from langchain.schema import Document
from langchain.memory.buffer import ConversationBufferMemory

# using Qdrant Vector store
# from qdrant_client import QdrantClient
# from langchain_community.vectorstores import Qdrant

# using Chroma Vector Store
# import chromadb
# from langchain_community.vectorstores import Chroma

# using PineCone Vector store
# from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec



from src.config.appconfig import GROQ_API_KEY, MODEL_NAME, TEMPERATURE, QDRANT_API_KEY, PINECONE_API_KEY
from src.tools.prompt_neck import PDF_SYS_PROMPT_NECK

# PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
# GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

import warnings;warnings.filterwarnings("ignore")

####---------------------------------------------RAG---------------------------------------------------------------####

class RetrievalAugmentGeneration:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 100):
        self.persist_directory = r"src\docs\chroma"
        self.llm = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY, temperature=TEMPERATURE, max_retries=5)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.groq_api_key = GROQ_API_KEY
        self.model_name = MODEL_NAME
        self.temperature = TEMPERATURE
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.initialize_session_state()
        self.embeddings = self.load_embeddings()
        self.vector_store = None
        self.loaded_doc = None


    @st.cache_resource
    def load_llm(_self):
        llm = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY, temperature=TEMPERATURE)
        return llm

#-----------------------------------------------Initialize Session State---------------------------------#
    def initialize_session_state(self):
        if "id" not in st.session_state:
            st.session_state.id = uuid.uuid4()
            st.session_state.file_cache = {}

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        if "messages" not in st.session_state:
            st.session_state.messages = []
    
    @staticmethod
    def display_pdf(file):
        with st.sidebar:
            st.markdown("### PDF Preview")
            file_content = file.read()
            base64_pdf = base64.b64encode(file_content).decode("utf-8")
            file.seek(0)  # Reset file pointer to the beginning

            pdf_display = f"""
                <iframe src="data:application/pdf;base64,{base64_pdf}" 
                height="300" type="application/pdf">
                </iframe>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)
            
            # Add download button
            st.download_button(
                label="Download PDF",
                data=file_content,
                file_name=file.name,
                mime="application/pdf"
            )
            
    
#-----------------------------------------------Load PDF Document---------------------------------#    
    def document_loader(_self, uploaded_file) -> Optional[List[Document]]:
        """
        Loads a PDF document from an uploaded file and splits it into pages.
        Args:
        uploaded_file (UploadedFile): Streamlit UploadedFile object.
        Returns:
        list: List of pages from the PDF document.
        """
        
        if uploaded_file is None:
            st.warning("Please upload a PDF file.")
            return None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            pdf_loader = PyPDFLoader(tmp_file_path)
            pages = pdf_loader.load_and_split()
            
            _self.loaded_doc = pages
            
            os.unlink(tmp_file_path)
            
            if not pages:
                st.error("No content found in the uploaded PDF. Please check the file.")
                return None
            
            logger.info(f"Successfully loaded {len(pages)} pages from the PDF.")
            return pages
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            st.error(f"Error loading PDF: {str(e)}")
            return None
     
    # def document_loader(self, uploaded_file):
    #     if uploaded_file is None:
    #         st.warning("Please upload a PDF file.")
    #         return None
        
    #     try:
    #         session_id = st.session_state.id
    #         with tempfile.TemporaryDirectory() as temp_dir:
    #             file_path = os.path.join(temp_dir, uploaded_file.name)
                
    #             with open(file_path, "wb") as f:
    #                 f.write(uploaded_file.getvalue())
                
    #             file_key = f"{session_id}-{uploaded_file.name}"
    #             st.write("Indexing your document...")

    #             if file_key not in st.session_state.get('file_cache', {}):
    #                 if os.path.exists(temp_dir):
    #                     loader = SimpleDirectoryReader(
    #                         input_dir=temp_dir,
    #                         required_exts=[".pdf"],
    #                         recursive=True
    #                     )
    #                 else:    
    #                     st.error('Could not find the file you uploaded, please check again...')
    #                     st.stop()

    #                 docs = loader.load_data()
    #                 # Convert llama_index documents to langchain documents
    #                 langchain_docs = [Document(page_content=doc.text, metadata=doc.metadata) for doc in docs]
    #                 self.loaded_doc = langchain_docs
    #                 return langchain_docs
        
    #     except Exception as e:
    #         st.error(f"An error occurred: {e}")
    #         st.stop()
        
#-----------------------------------------------Load Embeddings---------------------------------#    
    @st.cache_resource
    def load_embeddings(_self):
        try:
            # embed_model = HuggingFaceEmbedding(
            #     model_name="BAAI/bge-large-en-v1.5", 
            #     trust_remote_code=True
            # )
            # Settings.embed_model = embed_model
            
            embeddings =  HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
            
            # embeddings =  OpenAIEmbeddings(
            #     api_key=_self.openai_api_key,
            #     model="text-embedding-3-small",
            #     max_retries=3,
            #     dimensions=1536
            # )
            
            logger.info("Embeddings loaded successfully")
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            st.error(f"Error loading embeddings: {str(e)}")
            return None
    

#-----------------------------------------------Creating Vector Store with Chroma---------------------------------#       
    # @st.cache_resource
    # def create_vector_store(_self, _documents: List[Document]) -> Optional[Chroma]:
    #     if not _documents:
    #         logger.warning("No documents provided to the vector store.")
    #         return None

    #     try:
    #         text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=_self.chunk_size,
    #         chunk_overlap=_self.chunk_overlap,
    #         separators=["\n\n", "\n", " ", ""]
    #         )
    #         texts = text_splitter.split_documents(_documents)
            
    #         vector_store = Chroma.from_documents(
    #             documents=texts,
    #             embedding=_self.load_embeddings(),
    #             persist_directory=_self.persist_directory
    #         )
    #         vector_store.persist()
            
    #         logger.info(f"Vector store created with {len(texts)} chunks.")
    #         _self.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    #         return vector_store
        
    #     except Exception as e:
    #         logger.error(f"Error creating vector store: {str(e)}")
    #         st.error(f"Error creating vector store: {str(e)}")
    #         return None
        
    
    
    #-----------------------------------------------Creating Vector Store with PineCone---------------------------------#       
    # @st.cache_resource
    def create_vector_store(_self, _documents: List[Document]) -> Optional[PineconeVectorStore]:
        if not _documents:
            logger.warning("No documents provided to the vector store.")
            return None

        try:
            # Define the index name
            index_name = 'pdf2'

            # Initialize Pinecone client
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # List all existing indexes
            existing_indexes = pc.list_indexes()
            logger.info(f"List of indexes: {existing_indexes}")

            # Check if the index already exists
            # if index_name not in existing_indexes:
            #     logger.info(f"Creating new index: {index_name}")
            #     pc.create_index(
            #         name=index_name,
            #         dimension=768,
            #         metric="cosine",
            #         spec=ServerlessSpec(
            #             region="us-east-1",
            #             cloud="aws",
            #         )
            #     )
            #     # Wait until the index is ready
            #     while True:
            #         index_info = pc.describe_index(index_name)
            #         if index_info["status"]["ready"]:
            #             break
            #         logger.info("Waiting for index to be ready...")
            #         time.sleep(1)
            # else:
            #     logger.info(f"Index '{index_name}' already exists. Using the existing index.")

            # Connect to the existing index
            index = pc.Index(name=index_name)

            # Split the document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=_self.chunk_size,
                chunk_overlap=_self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            texts = text_splitter.split_documents(_documents)
            logger.info(f"Successfully split documents into {len(texts)} chunks.")

            # Create the Pinecone vector store
            vector_store = PineconeVectorStore.from_documents(
                documents=texts,
                index=index,
                embedding=_self.load_embeddings(),
                namespace="wondervector5000"
            )
            # Add new documents to the existing vector store
            vector_store.add_documents(texts)
            logger.info(f"Vector store created with {len(texts)} chunks.")

            # Set the retriever
            _self.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            return vector_store

        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            st.error(f"Error creating vector store: {str(e)}")
            return None
     
    
    #-----------------------------------------------Creating Vector Store with Qdrant---------------------------------#       
    # def create_vector_store(self, _documents: List[Document]) -> Optional[Qdrant]:
    #     if not _documents:
    #         logger.warning("No documents provided to the vector store.")
    #         return None

    #     try:
    #         text_splitter = RecursiveCharacterTextSplitter(
    #             chunk_size=self.chunk_size,
    #             chunk_overlap=self.chunk_overlap,
    #             separators=["\n\n", "\n", " ", ""]
    #         )
    #         texts = text_splitter.split_documents(_documents)
            
    #         from src.config.appconfig import QDRANT_API_KEY
    #         # Connect to Qdrant Cloud using API key and cloud URL
    #         qdrant_client = QdrantClient(
    #             url="https://c5ebb319-693e-4833-9fee-0dd76cd68e67.europe-west3-0.gcp.cloud.qdrant.io:6333",
    #             api_key=QDRANT_API_KEY
    #         )
            
    #         # Create vector store using Qdrant
    #         vector_store = Qdrant.from_documents(
    #             documents=texts,
    #             embedding=self.load_embeddings(),
    #             client=qdrant_client,
    #             collection_name="pdf-gpt"
    #         )
            
    #         logger.info(f"Vector store created with {len(texts)} chunks.")
    #         self.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    #         return vector_store

    #     except Exception as e:
    #         logger.error(f"Error creating vector store: {str(e)}")
    #         st.error(f"Error creating vector store: {str(e)}")
    #         return None

       
    
#-----------------------------------------------Creating Retriever---------------------------------# 
    def retriever(self, user_query: str) -> Dict[str, Any]:
        logger.info('Retrieving Relevant Documents')
        if os.path.exists(self.persist_directory) == False:
                self.document_loader()
                self.create_vector_store()
        
        if not self.vector_store:
            logger.error("Vector store not initialized.")
            return {"error": "Vector store not initialized"}
        
        try:
            
            # relevant_docs = self.retriever.get_relevant_documents(user_query)
            # relevant_docs = self.vector_store.similarity_search(user_query, k=4)
            relevant_docs = self.retriever.get_relevant_documents(user_query)
            
            relevant_docs = self.vector_store.similarity_search(user_query, k=4)
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents.")
            
            # return {"documents": relevant_docs}
            
            return {
                "relevant_content": [doc.page_content for doc in relevant_docs],
                "metadata": [doc.metadata for doc in relevant_docs]
            }
            
        except Exception as e:
            logger.error(f"Error in retriever: {str(e)}")
            return {"error": str(e)}
    
#-----------------------------------------------Format Docs---------------------------------#  
    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    
#-----------------------------------------------Creating RAG Pipeline---------------------------------# 
    def create_rag_pipeline(self):
        if not self.vector_store:
            logger.error("Vector store not initialized. Cannot create RAG application.")
            return None
        
        # retriever = _self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        
        prompt = PromptTemplate(input_variables=['context', 'input'], template=PDF_SYS_PROMPT_NECK)
        
        retriever_chain = (
            {"context": self.retriever | self.format_docs, "input": RunnablePassthrough()} 
            | prompt 
            | self.llm
            | StrOutputParser()
        )
        
        return retriever_chain
        

#-----------------------------------------------Chat with PDF---------------------------------# 
    def chat_with_pdf(self, user_query: str) -> Dict[str, str]:
        try:
            rag_chain = self.create_rag_pipeline()
            if not rag_chain:
                raise ValueError("RAG pipeline could not be created.")
            
            response = rag_chain.invoke(user_query)
            self.memory.chat_memory.add_user_message(user_query)
            self.memory.chat_memory.add_ai_message(response)
            st.session_state.chat_history.append(('Human', user_query))
            st.session_state.chat_history.append(('AI', response))
            logger.info("Successfully generated response.")
            return {'result': response}
        except Exception as e:
            error_message = f"Error in chat_with_pdf: {str(e)}"
            logger.error(error_message)
            return {'error': error_message}
    
    
    def clear_session(self):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.context = None
        self.memory.clear()
        self.vector_store = None
        self.loaded_doc = None
        self.current_file_name = None
        logging.info("Session cleared for new PDF upload.")
    
    
    def process_uploaded_file(self, uploaded_file):
        self.loaded_doc = self.document_loader(uploaded_file)
        if self.loaded_doc:
            self.vector_store = self.create_vector_store(self.loaded_doc)
            if self.vector_store:
                logger.info("Vector store initialized sucscessfully.")
            else:
                logger.error("Failed to create vector store.")
        else:
            logger.error("Failed to load the PDF. Please check the file and try again.")
    
    
    # def process_uploaded_file(self, uploaded_file):
    #     if uploaded_file.name != self.current_file_name:
    #         self.clear_session()
    #         self.current_file_name = uploaded_file.name

    #     self.loaded_doc = self.document_loader(uploaded_file)
    #     if self.loaded_doc:
    #         self.vector_store = self.create_vector_store(self.loaded_doc)
    #         if self.vector_store:
    #             logging.info("Vector store initialized successfully.")
    #             return True
    #         else:
    #             logging.error("Failed to create vector store.")
    #     else:
    #         logging.error("Failed to load the PDF.")
    #     return False
            

    def get_chat_history(self) -> List[tuple]:
        return st.session_state.chat_history
    
    @staticmethod
    def clear_chat_history():
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.context = None
        # self.memory.clear()
        gc.collect()
        logger.info("Chat history cleared and memory reset.")
        st.success("Conversation history cleared successfully!")
