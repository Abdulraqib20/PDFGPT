####---------------------------------------------Groq---------------------------------------------------------------####
import logging
from typing import Optional, List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv;load_dotenv()
from langchain_groq import ChatGroq
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
# from langchain_community.vectorstores import Pinecone as langchain_pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.schema import StrOutputParser, Document
import os
import tempfile
import streamlit as st



# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# GROQ_API_KEY=os.getenv("GROQ_API_KEY")

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# MODEL_NAME = "llama3-70b-8192"
# TEMPERATURE=0.2
# CHUNK_SIZE=700
# CHUNK_OVERLAP=100

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


####---------------------------------------------RAG---------------------------------------------------------------####

class RetrievalAugmentGeneration:
    def __init__(self, groq_api_key: str, pinecone_api_key: str, openai_api_key: str, 
                 model_name: str = "llama3-70b-8192", temperature: float = 0.2, 
                 chunk_size: int = 700, chunk_overlap: int = 100):
        self.groq_api_key = groq_api_key
        self.pinecone_api_key = pinecone_api_key
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.temperature = temperature
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
            logger.info("Embeddings loaded successfully")
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            st.error(f"Error loading embeddings: {str(e)}")
            return None
        
        # try:
        #     embeddings =  OpenAIEmbeddings(
        #         api_key=_self.openai_api_key,
        #         model="text-embedding-3-small",
        #         max_retries=3,
        #         dimensions=1536
        #     )
        #     logger.info("Embeddings loaded successfully")
            
        #     return embeddings
        
        # except Exception as e:
        #     logger.error(f"Error loading embeddings: {str(e)}")
        #     return None
    

#-----------------------------------------------Creating Vector Store---------------------------------#       
    @st.cache_resource
    def create_vector_store(_self, _documents: List[Document]) -> Optional[PineconeVectorStore]:
        if not _documents:
            logger.warning("No documents provided to the vector store.")
            return None
        
        try:
            pc = Pinecone(api_key=_self.pinecone_api_key)
            index_name = "pdf"
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=_self.chunk_size,
                chunk_overlap=_self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            texts = text_splitter.split_documents(_documents)
            
            
            # Create and return the vector store
            # embeddings = _self.load_embeddings()
            vector_store = PineconeVectorStore.from_documents(
                documents=texts,
                embedding=_self.load_embeddings(),
                index_name=index_name,
                namespace="pdf"
            )
            logger.info(f"Vector store created with {len(texts)} chunks.")
            return vector_store
        
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            st.error(f"Error creating vector store: {str(e)}")
            return None
        
    
#-----------------------------------------------Creating Retriever---------------------------------# 
    def retriever(self, user_query: str) -> Dict[str, Any]:
        logger.info('INSIDE RETRIEVER FUNCTION')
        
        if not self.vector_store:
            logger.error("Vector store not initialized.")
            return {"error": "Vector store not initialized"}
        
        try:
            
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  
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
        
        retriever = _self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})
        
        template_str = """
        You are an intelligent assistant specialized in analyzing, discussing and analyzing PDF documents. Your task is to provide 
        accurate, helpful, and concise responses to questions about the uploaded PDF file. Use only the information 
        provided in the context to answer the question.
        
        Provide detailed, data-driven insights based on the uploaded PDF File. If the answer is not explicitly stated in the context, use your 
        reasoning skills to provide the most likely answer based on the given information and suggest what additional information might be needed.

        Be thorough in your analysis, but do not invent or assume information that is not present in the data. 
        If you're unsure about any aspect, ask for clarification or more specific questions.

        Context from the PDF: {context}
        Human: {question}
        Assistant: Based on the provided context and your question, I'll analyze the PDF File and provide insights. 
        """
        
        prompt = PromptTemplate(input_variables=['context', 'input'], template=template_str)
        
        model = ChatGroq(model=_self.model_name, api_key=_self.groq_api_key, temperature=_self.temperature)
        
        retriever_chain = (
            {"context": retriever | _self.format_docs, "question": RunnablePassthrough()} 
            | prompt 
            | model
            | StrOutputParser()
        )
        
        return retriever_chain


#-----------------------------------------------Load PDF Document---------------------------------#
    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        

#-----------------------------------------------Chat with PDF---------------------------------# 
    def chat_with_pdf(self, user_query: str) -> Dict[str, str]:
        rag_chain = self.create_rag_pipeline()
        if not rag_chain:
            return {"error": "RAG pipeline could not be created."}
        
        try:
            response = rag_chain.invoke(user_query)
            st.session_state.chat_history.append(('Human', user_query))
            st.session_state.chat_history.append(('AI', response))
            logger.info("Successfully generated response.")
            return {'result': response}
        except Exception as e:
            logger.error(f"Error in chat_with_pdf: {str(e)}")
            return {'error': str(e)}
    
    
    def process_uploaded_file(self, uploaded_file):
        self.loaded_doc = self.document_loader(uploaded_file)
        if self.loaded_doc:
            self.vector_store = self.create_vector_store(self.loaded_doc)
            if self.vector_store:
                logger.info("Vector store initialized successfully.")
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

        
        
