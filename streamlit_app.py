# Import libraries
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from openai.error import RateLimitError
import logging

# Set up of logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Streamlit page
st.set_page_config(page_title="🎉 Hubert's Bachelor Party Assistant", page_icon="🎉")

# Initialize session state variables
def initialize_session_state():
    if 'authorized' not in st.session_state:
        st.session_state['authorized'] = False
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "assistant", "content": "Hey there! I’m happy to answer any questions you have about the trip :)"}
        ]

initialize_session_state()

# Function to authenticate the user
def authenticate_user():
    password = st.text_input("Type your password:", type="password")
    login_button = st.button("Log-in")

    if login_button:
        if password == st.secrets["bot_secrets"]["password"]:
            st.session_state['authorized'] = True
            st.success("Password correct!")
            logger.info("User logged in successfully.")
            st.experimental_rerun()
        else:
            st.error("Incorrect password. Please try again.")
            logger.warning("Failed login attempt.")
            st.stop()

# Show login interface if not authorized
if not st.session_state['authorized']:
    authenticate_user()
    st.stop()

# Once authorized, display the main title and description
st.title("🎉 Hubert's Bachelor Party Assistant")
st.write("Feel free to ask me anything about the trip! I'm here to help. 😊")

# Function to load Google Sheets and convert them to DataFrames
@st.cache_data
def load_sheets(client, sheet_names):
    dfs = []
    for name in sheet_names:
        try:
            sheet = client.open('Bachelor-party').worksheet(name)
            df = pd.DataFrame(sheet.get_all_records())
            dfs.append((df, name))
            logger.info(f"Loaded sheet: {name}")
        except Exception as e:
            st.error(f"Failed to load {name} sheet: {e}")
            logger.error(f"Error loading sheet {name}: {e}")
            dfs.append((pd.DataFrame(), name))
    return dfs

# Function to create documents from DataFrames
def create_documents(dfs):
    documents = []
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    for df, sheet_name in dfs:
        sheet_str = f"Sheet: {sheet_name}.\n"
        for _, row in df.iterrows():
            row_str = ' '.join(f"{col}: {row[col]}" for col in df.columns)
            sheet_str += row_str + '\n'
        # Split the document into chunks
        chunks = text_splitter.split_text(sheet_str)
        documents.extend(chunks)
    return documents

# Function to initialize the conversational AI chain
def initialize_qa_chain(openai_api_key, documents):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(documents, embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are Hubert's Bachelor Party Assistant. Answer questions based on the provided Google Sheets data with a friendly and informal tone.

Guidelines:
- Reference sheet names and column headers for context.
- If data is unavailable, politely indicate the lack of information.
- For packing lists, provide a comprehensive list from the 'packing_list' sheet.
- Avoid sensitive topics like racism, wars, homophobia, etc.
- Do not disclose personal or confidential information.
- Only use information present in the sheets without adding extra context.

Context: {context}

Question: {question}

Answer:
        """
    )

    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.4
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        memory=memory,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    return qa_chain

# Function to set up Google Sheets client
def setup_google_sheets():
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    credentials_info = st.secrets["google_credentials"]
    creds = Credentials.from_service_account_info(credentials_info, scopes=scope)
    client = gspread.authorize(creds)
    return client

# Load Google Sheets data
client = setup_google_sheets()
sheet_names = ['packing_list', 'schedule', 'costs', 'Q&A']
dfs = load_sheets(client, sheet_names)

# Create documents for the AI
documents = create_documents(dfs)

# Initialize the QA chain
openai_api_key = st.secrets["OPENAI_API_KEY"]
qa = initialize_qa_chain(openai_api_key, documents)

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and generate responses
if prompt := st.chat_input("Ask a question about Hubert's bachelor party:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Looking for an answer..."):
        try:
            response = qa({"question": prompt})
            answer = response['answer']
            sources = response.get('source_documents', [])

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

        except RateLimitError:
            st.error("You've exceeded the OpenAI API rate limit. Please check your plan and billing details.")
            logger.warning("Rate limit exceeded.")
        except Exception as e:
            st.error("An unexpected error occurred. Please try again later.")
            logger.error(f"Unexpected error: {e}")

# Button to clear chat history
if st.button("Clear chat history"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey there! I’m happy to answer any questions you have about the trip :)"}
    ]
    st.experimental_rerun()
