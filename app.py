import streamlit as st
import streamlit.components.v1 as components
import json
from typing import List, Any
import re
import logging
from dotenv import load_dotenv;load_dotenv()
import warnings;warnings.filterwarnings("ignore")

from pydantic import BaseModel, Field
# from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.exceptions import LangChainException

#-----------------------------------Set up Logging-------------------------------
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#-----------------------------------Importing the PDF-GPT Prompt Head-------------------------------
from src.tools.prompt_head import PDF_SYS_PROMPT_HEAD


from RAG import RetrievalAugmentGeneration

# PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
# GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

#---------------------------------------------Create a Streamlit app-----------------------------------------------
st.set_page_config(
    page_title="PDF-GPT",
    page_icon="📚",
    layout="centered",
    # initial_sidebar_state="collapsed"
)


# #-------------------------------------------------------Styling-------------------------------------------------
# st.markdown(
#     """
#     <style>
#         /* General */
#         body {
#             font-family: 'Karla', sans-serif;
#             color: #333;
#             background-color: #f4f4f9;
#         }

#         /* Header */
#         .main-header {
#             color: white;
#             padding: 20px;
#             text-align: center;
#             border-radius: 10px;
#             box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
#             margin-bottom: 20px;
#         }

#         .main-header h1 {
#             font-size: 2.5rem;
#             margin: 0;
#         }

#         /* Rocket emoji animation */
#         .main-header h1 span {
#             display: inline-block;
#             animation: rocket-animation 2s linear infinite;
#         }

#         @keyframes rocket-animation {
#             0%, 100% { transform: translateY(0); }
#             50% { transform: translateY(-10px); }
#         }

#         /* WhatsApp icon */
#         .whatsapp-icon {
#             height: 50px;
#             margin-right: 15px;
#             vertical-align: middle;
#         }

#         /* Intro and get started sections */
#         .section {
#             # background-color: #fff;
#             border: 1px solid #ddd;
#             padding: 25px;
#             border-radius: 10px;
#             box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
#             margin-bottom: 20px;
#         }

#         .section h2, .section h3 {
#             color: #007bff;
#             margin-bottom: 15px;
#         }

#         .section p {
#             line-height: 1.6;
#         }

#         /* Expander header */
#         .stExpanderHeader {
#             color: white;
#             padding: 10px 15px;
#             font-weight: bold;
#             border-radius: 5px;
#             cursor: pointer;
#         }

#         .stExpanderContent p {
#             margin-top: 0;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # st.title("AI Chat with PDF 💬")
# st.title("PDF-GPT 💬")

# ##-----------------------------------------------STYLE HEADER AND ABOUT SECTIONS--------------------------------------

# # Custom HTML, CSS, and JavaScript for animated tabs
# custom_html = """
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

# .tabs-container {
#     font-family: 'Karla', sans-serif;
#     max-width: 800px;
#     margin: 0 auto;
#     padding: 20px;
#     background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
#     border-radius: 20px;
#     box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
# }

# .tab-buttons {
#     display: flex;
#     justify-content: center;
#     margin-bottom: 20px;
# }

# .tab-button {
#     background: none;
#     border: none;
#     padding: 10px 20px;
#     font-size: 18px;
#     font-weight: 500;
#     color: #333;
#     cursor: pointer;
#     transition: all 0.3s ease;
#     position: relative;
#     overflow: hidden;
# }

# .tab-button::after {
#     content: '';
#     position: absolute;
#     bottom: 0;
#     left: 0;
#     width: 100%;
#     height: 3px;
#     background-color: #25D366;
#     transform: scaleX(0);
#     transition: transform 0.3s ease;
# }

# .tab-button:hover::after,
# .tab-button.active::after {
#     transform: scaleX(1);
# }

# .tab-content {
#     background: rgba(255, 255, 255, 0.8);
#     padding: 20px;
#     border-radius: 15px;
#     box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#     transition: all 0.5s ease;
#     opacity: 0;
#     transform: translateY(20px);
#     display: none;
# }

# .tab-content.active {
#     opacity: 1;
#     transform: translateY(0);
#     display: block;
# }

# .feature-list {
#     list-style-type: none;
#     padding: 0;
# }

# .feature-list li {
#     margin-bottom: 10px;
#     padding-left: 25px;
#     position: relative;
# }

# .feature-list li::before {
#     content: '🚀';
#     position: absolute;
#     left: 0;
#     top: 0;
# }

# </style>

# <div class="tabs-container">
#     <div class="tab-buttons">
#         <button class="tab-button active" onclick="showTab('how-to-use')">How To Use</button>
#     </div>
    
#     <div id="how-to-use" class="tab-content active">
#         <h2> ✨ How To Use</h2>
#         <p>
#             Welcome to the AI-Chat with PDF App developed by raqibcodes. The app allows you to unlock insights from your PDF, it is like using ChatGPT directly on your PDF. <br><br> You can chat with the PDF directly using the generative AI capabilities incorporated into the app!
#             Here's your guide to unlocking insights:
#         </p><br>
#         <ol>
#             <li><strong>Upload a PDF File:</strong> From the Sidebar Uploader, click the "Upload PDF" button </li>
#             <li><strong>AI Chat:</strong> Use the Generative AI feature to have a conversation with your uploaded PDF!</li>
#         </ol>
#     </div> 
# </div>

# <script>
# function showTab(tabId) {
#     // Hide all tab contents
#     var tabContents = document.getElementsByClassName('tab-content');
#     for (var i = 0; i < tabContents.length; i++) {
#         tabContents[i].classList.remove('active');
#     }
    
#     // Show the selected tab content
#     document.getElementById(tabId).classList.add('active');
    
#     // Update active state of tab buttons
#     var tabButtons = document.getElementsByClassName('tab-button');
#     for (var i = 0; i < tabButtons.length; i++) {
#         tabButtons[i].classList.remove('active');
#     }
#     event.currentTarget.classList.add('active');
# }
# </script>
# """

# # Render the custom HTML
# components.html(custom_html, height=460, scrolling=True)



# Define a color scheme
PRIMARY_COLOR = "#4A90E2"
SECONDARY_COLOR = "#F5A623"
BACKGROUND_COLOR = "#F0F4F8"
TEXT_COLOR = "#333333"

# Custom CSS
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {{
        font-family: 'Roboto', sans-serif;
        # background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
    }}
    
    .stApp {{
        max-width: 1200px;
        margin: 0 auto;
    }}

    h1, h2, h3 {{
        color: {PRIMARY_COLOR};
    }}

    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }}

    .stButton>button:hover {{
        background-color: {SECONDARY_COLOR};
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}

    .stTextInput>div>div>input {{
        border-radius: 5px;
        border: 1px solid #E0E0E0;
    }}

    .stFileUploader>div {{
        border-radius: 5px;
        border: 2px dashed {PRIMARY_COLOR};
        padding: 2rem;
    }}

    .stFileUploader>div:hover {{
        border-color: {SECONDARY_COLOR};
    }}

    .css-145kmo2 {{
        border: none;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}

    .css-1d391kg {{
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# App title and description
st.title("📚 PDF-GPT Chat")
st.markdown("Unlock insights from your PDF documents with AI-powered chat.")

# upload a PDF file
with st.sidebar:
    # How to Use
    st.sidebar.header("🚀 How to Use")
    st.sidebar.markdown("""
    1. **Upload your PDF**: Use the sidebar to upload your PDF document.
    2. **Start chatting**: Once your document is uploaded, use the chat interface to ask questions.
    3. **Explore insights**: The AI will analyze your document and provide relevant answers.
    """)
    
    st.header("📁 PDF Document Upload")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])




#-------------------------------------Pydantic model for PDF Chat--------------------------------
class PdfChat(BaseModel):
    """
    Input schema for analyzing PDF Document.
    """
    query: str = Field(..., description='Query about the PDF Document')

#-----------------------------------------------Input validation---------------------------------
# def validate_user_input(user_input: str) -> bool:
#     if not user_input or not re.match(r'^[a-zA-Z0-9\s\.\,\?\!]+$', user_input):
#         st.warning("Please enter a valid query using alphanumeric characters and basic punctuation.")
#         return False
#     return True
   
#----------------------------------------------------Function to process chat
def process_chat(model, user_input: str, chat_history: List[Any]) -> Any:
    return model.invoke({
        'chat_history': chat_history,
        'input': user_input
    })


#------------------------------------------------MAIN APPLICATION-----------------------------------------------------
def main():
    #------------------------------------------------------Initialize RAG-----------------------------------------
    rag = RetrievalAugmentGeneration()  
    rag.initialize_session_state()

    if uploaded_file is not None:
        
        # display pdf overview
        rag.display_pdf(uploaded_file)
        
        # Process uploaded file
        with st.spinner("Processing the uploaded PDF..."):
            docs = rag.document_loader(uploaded_file)
            if docs:
                rag.vector_store = rag.create_vector_store(docs)
            # rag.process_uploaded_file(uploaded_file)
            # rag.vector_store = rag.create_vector_store(rag.loaded_doc)    
 
        if rag.vector_store:
            st.sidebar.success("PDF processed and vector store created successfully!")
        else:
            st.sidebar.error("Failed to process PDF or create vector store. Please try again.")
            return

        #---------------------------------------CONVERTING TO FUNCTION DECLARATION OBJECT---------------------------------
        pdf_chats_func = convert_to_openai_function(PdfChat)

        #---------------------------------------------Create Chat Template----------------------------------------------

        prompt = ChatPromptTemplate.from_messages(
            [
                ('system', PDF_SYS_PROMPT_HEAD),
                MessagesPlaceholder(variable_name='chat_history'),
                HumanMessagePromptTemplate.from_template('{input}')
            ]
        )

        
        #---------------------------------------------Initialize Chat Models--------------------------------------
        chat_model = rag.load_llm()
        chat_with_tools = chat_model.bind_tools(tools=[pdf_chats_func]) 
        chain =  prompt | chat_with_tools

        #-----------------------------------------Display Chat History--------------------------------
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])

        #------------------------------------------User input
        if user_input := st.chat_input('Message PDF-GPT 💬', key='user_input'):
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(user_input)
        
            with st.chat_message('assistant', avatar="🤖"):
                try:

                    response = process_chat(chain, user_input ,st.session_state.chat_history)
                    st.session_state.chat_history.append(HumanMessage(content=user_input))
                    st.session_state.chat_history.append(response)
                    
                    logger.info(f"Initial AI response: {response}")
                
                    if response.content:
                        st.markdown(response.content)
                        st.session_state.messages.append({"role": "assistant", "content": response.content})
                    
                    # print(f"Response: {response}, '\n")
                    # print(f"Response Additional Kwargs: {response.additional_kwargs}, '\n")
                    
                    elif response.additional_kwargs.get('tool_calls'):
                        # Extracting information
                        tool_calls = response.additional_kwargs.get('tool_calls', [])
                        for call in tool_calls:
                            #  EXTRACT THE PARAMETERS TO BE PASSED TO THE FUNCTIONS
                            function = call.get('function', {})
                            function_name = function.get('name')
                            function_args = json.loads(function.get('arguments', '{}'))

                            # PERFORMS THE FUNCTION CALL OUTSIDE THE LLM MODEL
                            if function_name == 'PdfChat':
                                with st.status('Analyzing PDF...', expanded=True) as status:
                                    api_response = rag.retriever(function_args.get('query', ''))
                                    status.update(label='Analysis Complete', state='complete', expanded=False)
                                
                                # logger.info(f"API response: {api_response}")
                                
                                # PARSE THE RESPONSE OF THE API CALLS BACK INTO THE MODEL
                                tool_message = ToolMessage(content=str(api_response), name=function_name, tool_call_id=call.get('id'))
                                ai_response = process_chat(chain, str(tool_message), st.session_state.chat_history)
                                
                                logger.info(f"Final AI response: {ai_response}")

                            if ai_response.content:
                                # APPEND THE FUNCTION RESPONSE AND AI RESPONSE TO BOTH THE CHAT_HISTORY AND MESSAGE HISTORY FOR STREAMLIT 
                                st.markdown(ai_response.content)
                                st.session_state.chat_history.extend([tool_message, ai_response])
                                st.session_state.messages.append({"role": "assistant", "content": ai_response.content})
                            else:
                                st.warning("The AI didn't provide a clear response. Here's what I found in the document:")
                                st.json(api_response)
                                st.error("The AI didn't provide a response. Please try again.")

                    else:
                        st.warning("The AI couldn't generate a response based on the document. Please try rephrasing your question.")
                
                    st.rerun()
                    
                
                
                except (Exception,LangChainException) as e:
                    logger.error(f"An error occurred: {e}")
                    st.error("An unexpected error occurred. Please try again later.")
        
        #------------------------------------------Clear Conversation------------------------------------------
        if st.sidebar.button('Clear Conversation'):
            rag.clear_chat_history()
            st.rerun()
        
    else:
        st.title("")
        st.info("👈 To get started with PDF-GPT, please upload a PDF Document!")

if __name__ == "__main__":
    main()

# #---------------------------------------------------------FOOTER------------------------------------------------

st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        text-align: center;
    }
    .footer a {
        color: #007bff;
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

# Footer content
st.markdown('<div class="footer">Developed with ❤️ by <a href="https://github.com/Abdulraqib20" target="_blank">raqibcodes</a></div>', unsafe_allow_html=True)


# st.markdown(
#     """
#     <style>
#         @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;600&display=swap');

#         .footer-container {
#             font-family: 'Raleway', sans-serif;
#             margin-top: 50px;
#             padding: 30px 0;
#             width: 100vw;
#             position: absolute;
#             left: 50%;
#             right: 50%;
#             margin-left: -50vw;
#             margin-right: -50vw;
#             # overflow: hidden;
#         }

#         .footer-content {
#             display: flex;
#             justify-content: center;
#             align-items: center;
#             position: relative;
#             z-index: 2;
#         }

#         .footer-text {
#             color: #ffffff;
#             font-size: 20px;
#             font-weight: 300;
#             text-align: center;
#             margin: 0;
#             padding: 0 20px;
#             position: relative;
#         }

#         .footer-link {
#             color: #075E54;  /* WhatsApp dark green */
#             font-weight: 600;
#             text-decoration: none;
#             position: relative;
#             transition: all 0.3s ease;
#             padding: 5px 10px;
#             border-radius: 5px;
#         }

#         .footer-link:hover {
#             background-color: rgba(7, 94, 84, 0.1);  /* Slightly darker on hover */
#             box-shadow: 0 0 15px rgba(7, 94, 84, 0.2);
#         }

#         .footer-heart {
#             display: inline-block;
#             color: #FF0000;  /* Red heart */
#             font-size: 35px;
#             animation: pulse 1.5s ease infinite;
#         }

#         @keyframes pulse {
#             0% { transform: scale(1); }
#             50% { transform: scale(1.1); }
#             100% { transform: scale(1); }
#         }
#     </style>

#     <div class="footer-container">
#         <div class="footer-content">
#             <p class="footer-text">
#                 Developed by <a href="https://github.com/Abdulraqib20" target="_blank" class="footer-link">raqibcodes</a>
#             </p>
#         </div>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

