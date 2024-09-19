####---------------------------------------------RAG Pipeline-----------------------------------------------####

import os
import tempfile
import streamlit as st
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Optional, List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv;load_dotenv()
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores import Pinecone as PineConeVector
# from pinecone import Pinecone
# from langchain_pinecone import PineconeVectorStore
# from langchain_community.vectorstores import Pinecone as langchain_pinecone
from langchain.schema import StrOutputParser, Document
from langchain.memory.buffer import ConversationBufferMemory
import warnings;warnings.filterwarnings("ignore")

from src.config.appconfig import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from src.tools.prompt_neck import PDF_SYS_PROMPT_NECK

# PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
# GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


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


#-----------------------------------------------Initialize Session State---------------------------------#
    @staticmethod
    def initialize_session_state():
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = [] 

        if "messages" not in st.session_state:
            st.session_state.messages = []
            
    
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
            
        
#-----------------------------------------------Load Embeddings---------------------------------#    
    @st.cache_resource
    def load_embeddings(_self):
        try:
            embeddings =  HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
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
    

#-----------------------------------------------Creating Vector Store---------------------------------#       
    @st.cache_resource
    def create_vector_store(_self, _documents: List[Document]) -> Optional[Chroma]:
        if not _documents:
            logger.warning("No documents provided to the vector store.")
            return None

        try:
            text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=_self.chunk_size,
            chunk_overlap=_self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
            )
            texts = text_splitter.split_documents(_documents)
            
            embeddings = _self.load_embeddings()
            
            vector_store = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=_self.persist_directory
            )
            vector_store.persist()
            
            logger.info(f"Vector store created with {len(texts)} chunks.")
            return vector_store
        
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            st.error(f"Error creating vector store: {str(e)}")
            return None
        
    
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
            
            k = 4
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}  
            )
            
            relevant_docs = self.vector_store.similarity_search(user_query, k=4)
            # relevant_docs = retriever.get_relevant_documents(user_query)
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents.")
            
            # return {"documents": relevant_docs}
            
            return {
                "relevant_content": [doc.page_content for doc in relevant_docs],
                "metadata": [doc.metadata for doc in relevant_docs]
            }
            
        except Exception as e:
            logger.error(f"Error in retriever: {str(e)}")
            return {"error": str(e)}
    
    
#-----------------------------------------------Creating RAG Pipeline---------------------------------# 
    def create_rag_pipeline(_self):
        if not _self.vector_store:
            logger.error("Vector store not initialized. Cannot create RAG application.")
            return None
        
        k = 4
        retriever = _self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        
        prompt = PromptTemplate(input_variables=['context', 'input'], template=PDF_SYS_PROMPT_NECK)
        
        retriever_chain = (
            {"context": retriever | _self.format_docs, "input": RunnablePassthrough()} 
            | prompt 
            | _self.llm
            | StrOutputParser()
        )
        
        return retriever_chain


#-----------------------------------------------Load PDF Document---------------------------------#
    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        

#-----------------------------------------------Chat with PDF---------------------------------# 
    def chat_with_pdf(self, user_query: str) -> Dict[str, str]:
        try:
            rag_chain = self.create_rag_pipeline()
            if not rag_chain:
                raise ValueError("RAG pipeline could not be created.")
            
            response = rag_chain.invoke(user_query)
            st.session_state.chat_history.append(('Human', user_query))
            st.session_state.chat_history.append(('AI', response))
            logger.info("Successfully generated response.")
            return {'result': response}
        except Exception as e:
            error_message = f"Error in chat_with_pdf: {str(e)}"
            logger.error(error_message)
            return {'error': error_message}
    
    
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
            

    def get_chat_history(self) -> List[tuple]:
        return st.session_state.chat_history
    

    def clear_chat_history(self):
        st.session_state.chat_history = []
        st.session_state.messages = []
        logger.info("Chat history cleared.")
        st.success("Conversation history cleared successfully!")
