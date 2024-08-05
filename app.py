import streamlit as st
import streamlit.components.v1 as components
from io import BytesIO
from langchain_groq.chat_models import ChatGroq
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.exceptions import LangChainException
import os
import json
from typing import List, Dict, Any
from functools import wraps
import re
import logging
from dotenv import load_dotenv;load_dotenv()
import warnings;warnings.filterwarnings("ignore")

from RAG import RetrievalAugmentGeneration

#---------------------------------------------Create a Streamlit app-----------------------------------------------
st.set_page_config(
    page_title="PDF Analysis",
    page_icon="📊",
    layout="wide",
    # initial_sidebar_state="collapsed"
)


#-------------------------------------------------------Styling-------------------------------------------------
st.markdown(
    """
    <style>
        /* General */
        body {
            font-family: 'Karla', sans-serif;
            color: #333;
            background-color: #f4f4f9;
        }

        /* Header */
        .main-header {
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .main-header h1 {
            font-size: 2.5rem;
            margin: 0;
        }

        /* Rocket emoji animation */
        .main-header h1 span {
            display: inline-block;
            animation: rocket-animation 2s linear infinite;
        }

        @keyframes rocket-animation {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }

        /* WhatsApp icon */
        .whatsapp-icon {
            height: 50px;
            margin-right: 15px;
            vertical-align: middle;
        }

        /* Intro and get started sections */
        .section {
            # background-color: #fff;
            border: 1px solid #ddd;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .section h2, .section h3 {
            color: #007bff;
            margin-bottom: 15px;
        }

        .section p {
            line-height: 1.6;
        }

        /* Expander header */
        .stExpanderHeader {
            color: white;
            padding: 10px 15px;
            font-weight: bold;
            border-radius: 5px;
            cursor: pointer;
        }

        .stExpanderContent p {
            margin-top: 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# st.title("AI Chat with PDF 💬")
st.title("PDF-GPT 💬")

##-----------------------------------------------STYLE HEADER AND ABOUT SECTIONS--------------------------------------

# Custom HTML, CSS, and JavaScript for animated tabs
custom_html = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

.tabs-container {
    font-family: 'Karla', sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
}

.tab-buttons {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}

.tab-button {
    background: none;
    border: none;
    padding: 10px 20px;
    font-size: 18px;
    font-weight: 500;
    color: #333;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.tab-button::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background-color: #25D366;
    transform: scaleX(0);
    transition: transform 0.3s ease;
}

.tab-button:hover::after,
.tab-button.active::after {
    transform: scaleX(1);
}

.tab-content {
    background: rgba(255, 255, 255, 0.8);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.5s ease;
    opacity: 0;
    transform: translateY(20px);
    display: none;
}

.tab-content.active {
    opacity: 1;
    transform: translateY(0);
    display: block;
}

.feature-list {
    list-style-type: none;
    padding: 0;
}

.feature-list li {
    margin-bottom: 10px;
    padding-left: 25px;
    position: relative;
}

.feature-list li::before {
    content: '🚀';
    position: absolute;
    left: 0;
    top: 0;
}

</style>

<div class="tabs-container">
    <div class="tab-buttons">
        <button class="tab-button active" onclick="showTab('how-to-use')">How To Use</button>
    </div>
    
    <div id="how-to-use" class="tab-content active">
        <h2> ✨ How To Use</h2>
        <p>
            Welcome to the AI-Chat with PDF App developed by raqibcodes. The app allows you to unlock insights from your PDF, it is like using ChatGPT directly on your PDF. <br><br> You can chat with the PDF directly using the generative AI capabilities incorporated into the app!
            Here's your guide to unlocking insights:
        </p><br>
        <ol>
            <li><strong>Upload a PDF File:</strong> From the Sidebar Uploader, click the "Upload PDF" button </li>
            <li><strong>AI Chat:</strong> Use the Generative AI feature to have a conversation with your uploaded PDF!</li>
        </ol>
    </div> 
</div>

<script>
function showTab(tabId) {
    // Hide all tab contents
    var tabContents = document.getElementsByClassName('tab-content');
    for (var i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove('active');
    }
    
    // Show the selected tab content
    document.getElementById(tabId).classList.add('active');
    
    // Update active state of tab buttons
    var tabButtons = document.getElementsByClassName('tab-button');
    for (var i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove('active');
    }
    event.currentTarget.classList.add('active');
}
</script>
"""

# Render the custom HTML
components.html(custom_html, height=460, scrolling=True)



# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# GROQ_API_KEY=os.getenv("GROQ_API_KEY")

PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
MODEL_NAME = "llama3-70b-8192"
TEMPERATURE=0.2


#----------------------------------------------------Set up Logging --------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


#-------------------------------------Pydantic model for PDF Chat--------------------------------
class PdfChat(BaseModel):
    """
    Input schema for analyzing PDF Document.
    """
    query: str = Field(..., description='Query about the PDF Document')

#-----------------------------------------------Input validation---------------------------------
def validate_user_input(user_input: str) -> bool:
    if not user_input or not re.match(r'^[a-zA-Z0-9\s\.\,\?\!]+$', user_input):
        st.warning("Please enter a valid query using alphanumeric characters and basic punctuation.")
        return False
    return True
   
#----------------------------------------------------Function to process chat
def process_chat(model, user_input: str, chat_history: List[Any]) -> Any:
    return model.invoke({
        'chat_history': chat_history,
        'input': user_input
    })
    

#---------------------------------------------------Conversation manager------------------------------
class ConversationManager:
    def __init__(self):
        self.conversations = {}
    
    def start_conversation(self, user_id):
        self.conversations[user_id] = []
    
    def add_message(self, user_id, role, content):
        if user_id not in self.conversations:
            self.start_conversation(user_id)
        self.conversations[user_id].append({"role": role, "content": content})
    
    def get_conversation(self, user_id):
        return self.conversations.get(user_id, [])


#------------------------------------------------MAIN APPLICATION-----------------------------------------------------
def main():
    conversation_manger = ConversationManager()
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        
        #------------------------------------------------------Initialize RAG-----------------------------------------
        rag = RetrievalAugmentGeneration(
            groq_api_key=GROQ_API_KEY,
            pinecone_api_key=PINECONE_API_KEY,
            openai_api_key=OPENAI_API_KEY
        )
        
        rag.process_uploaded_file(uploaded_file)
    

        #---------------------------------------CONVERTING TO FUNCTION DECLARATION OBJECT---------------------------------
        pdf_chats_func = convert_to_openai_function(PdfChat)

        #---------------------------------------------Create Chat Template----------------------------------------------
        system_prompt_template = """
        You are an AI assistant specialized in analyzing PDF documents. 
        Use the provided tools to answer questions about the uploaded PDF.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ('system', system_prompt_template),
                MessagesPlaceholder(variable_name='chat_history'),
                HumanMessagePromptTemplate.from_template('{input}')
            ]
        )

        
        #---------------------------------------------Initialize Chat Models--------------------------------------
        
        chat_model = ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME, temperature=TEMPERATURE, max_retries=5)
        chat_with_tools = chat_model.bind_tools(tools=[pdf_chats_func]) 
        chain =  prompt | chat_with_tools

        #---------------------------------------------Initialize Session State--------------------------------
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = [] 

        if "messages" not in st.session_state:
            st.session_state.messages = []

        #-----------------------------------------Display Chat History--------------------------------
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])

        #------------------------------------------User input
        if user_input := st.chat_input('Message PDF-GPT 💬', key='user_input'):
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(user_input)
        
        

            with st.chat_message('assistant', avatar="🤖"):
                try:

                    response = process_chat(chain,user_input,st.session_state.chat_history)
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
                                st.error("The AI didn't provide a response. Please try again.")

                    else:
                        st.error("The AI response was empty and no tool calls were made. Please try again.")
                
                    st.rerun()
                    
                
                
                except (Exception,LangChainException) as e:
                    logger.error(f"An error occurred: {e}")
        
        #------------------------------------------Clear Conversation------------------------------------------
        if st.sidebar.button('Clear Conversation'):
            rag.clear_chat_history()
        
    else:
        st.warning("Couldn't proceed without uploading a PDF file. Please upload a PDF file!")
        rag = None
 


if __name__ == "__main__":
    main()
    

#---------------------------------------------------------FOOTER------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;600&display=swap');

        .footer-container {
            font-family: 'Raleway', sans-serif;
            margin-top: 50px;
            padding: 30px 0;
            width: 100vw;
            position: absolute;
            left: 50%;
            right: 50%;
            margin-left: -50vw;
            margin-right: -50vw;
            # overflow: hidden;
        }

        .footer-content {
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            z-index: 2;
        }

        .footer-text {
            color: #ffffff;
            font-size: 20px;
            font-weight: 300;
            text-align: center;
            margin: 0;
            padding: 0 20px;
            position: relative;
        }

        .footer-link {
            color: #075E54;  /* WhatsApp dark green */
            font-weight: 600;
            text-decoration: none;
            position: relative;
            transition: all 0.3s ease;
            padding: 5px 10px;
            border-radius: 5px;
        }

        .footer-link:hover {
            background-color: rgba(7, 94, 84, 0.1);  /* Slightly darker on hover */
            box-shadow: 0 0 15px rgba(7, 94, 84, 0.2);
        }

        .footer-heart {
            display: inline-block;
            color: #FF0000;  /* Red heart */
            font-size: 35px;
            animation: pulse 1.5s ease infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
    </style>

    <div class="footer-container">
        <div class="footer-content">
            <p class="footer-text">
                Developed by <a href="https://github.com/Abdulraqib20" target="_blank" class="footer-link">raqibcodes</a>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

