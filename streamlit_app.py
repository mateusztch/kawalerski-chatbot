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
import openai  

# Website config
st.set_page_config(page_title="🎉 Hubert's Bachelor Party Assistant", page_icon="🎉")

# Authorization status tracking
if 'authorized' not in st.session_state:
    st.session_state['authorized'] = False

# Password input if not authorized
if not st.session_state['authorized']:
    password = st.text_input("Type your password:", type="password")
    login_button = st.button("Log-in")

    if login_button:
        if password == st.secrets["bot_secrets"]["password"]:
            st.session_state['authorized'] = True
            st.success("Password correct!")
            st.experimental_rerun()  
        else:
            st.error("Error. Try again later.")
            st.stop()  
    else:
        st.stop()  

# Initial setup - if authorized
st.title("🎉 Hubert's Bachelor Party Assistant")
st.write(
    "Don't ask too many questions because I'm paying for API calls 🤪"
)

# Import openai API
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Installation of LLM
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo",
    temperature=0.3  
)

# Google Sheets API
scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
credentials_info = st.secrets["google_credentials"]
creds = Credentials.from_service_account_info(credentials_info, scopes=scope)
client = gspread.authorize(creds)

# Google Sheets data
spreadsheet = client.open('Bachelor-party')
sheet1 = spreadsheet.worksheet('packing_list')
sheet2 = spreadsheet.worksheet('schedule')
sheet3 = spreadsheet.worksheet('costs')
sheet4 = spreadsheet.worksheet('Q&A')

# Conversion sheets to DataFrames
df1 = pd.DataFrame(sheet1.get_all_records())
df2 = pd.DataFrame(sheet2.get_all_records())
df3 = pd.DataFrame(sheet3.get_all_records())
df4 = pd.DataFrame(sheet4.get_all_records())
dfs = [(df1, 'packing_list'), (df2, 'schedule'), (df3, 'costs'), (df4, 'Q&A')]

# List of docs
documents = []

# Iterate over each DF
for df, sheet_name in dfs:
    sheet_str = f"Arkusz: {sheet_name}.\n"
    for index, row in df.iterrows():
        row_str = ' '.join(f"{col_name}: {row[col_name]}" for col_name in df.columns)
        sheet_str += row_str + '\n'
    documents.append(sheet_str)

# Create chunks 
# 2000-char chunks (~500 tokens) with 400-char overlap ensure context continuity and fit well within GPT's 4096-token limit, leaving room for prompts and responses.

text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
texts = []
for doc in documents:
    texts.extend(text_splitter.split_text(doc))

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_texts(texts, embeddings)

# Create chate memory memory
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)

# Prompt template
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

# Chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
    memory=memory,
    verbose=False,
    combine_docs_chain_kwargs={"prompt": prompt_template}
)

# Conversation history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hey there! I’m happy to answer any questions you have about the trip :)"}]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about Hubert's bachelor party:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Chain response
    with st.spinner("Workin on it!"):
        try:
            response = qa({"question": prompt})
            answer = response['answer']
            sources = response.get('source_documents', [])

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

        except RateLimitError:
            st.error("You've exceeded the OpenAI API rate limit. Check your plan and billing details.")
        except Exception as e:
            st.error("An unexpected error occurred while processing your question. Please try again later.")

# Clear chat history button
if st.button("Clear chat history"):
    st.session_state.messages = []
    st.experimental_rerun()
