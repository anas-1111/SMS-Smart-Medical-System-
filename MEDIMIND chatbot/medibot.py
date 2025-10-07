import os
import streamlit as st

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryBufferMemory,
    CombinedMemory
)

load_dotenv()

# === UI THEME ===
PRIMARY = "#22d3ee"
PRIMARY_DARK = "#06b6d4"
BG_START = "#90c450"
BG_END = "#194bbe"
CARD_BG = "rgba(17, 24, 39, 0.7)"
USER_BUBBLE = "#d35624"
ASSIST_BUBBLE = "#1394c3"
TEXT = "#e5e7eb"
RADIUS = "16px"

st.set_page_config(page_title="Medical Chatbot NTI", page_icon="🩺", layout="centered")

# === Custom CSS ===
st.markdown(
    f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] {{
        background: linear-gradient(145deg, {BG_START} 0%, {BG_END} 100%) !important;
        color: {TEXT};
    }}
    [data-testid="stAppViewBlockContainer"] {{
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
        max-width: 860px;
    }}
    h1, h2, h3, .stMarkdown h1 {{
        color: {TEXT} !important;
    }}
    [data-testid="stChatInput"] textarea {{
        background: {CARD_BG} !important;
        color: {TEXT} !important;
        border-radius: {RADIUS} !important;
    }}
    [data-testid="stChatInput"] button[kind="primary"] {{
        background: {PRIMARY} !important;
        color: #041014 !important;
        border-radius: 12px !important;
    }}
    [data-testid="stChatInput"] button[kind="primary"]:hover {{
        background: {PRIMARY_DARK} !important;
    }}
    .stChatMessage[data-testid="chat-message-user"] > div:nth-child(2) {{
        background: {USER_BUBBLE} !important;
        color: {TEXT} !important;
        border-radius: {RADIUS};
        padding: 14px 16px;
    }}
    .stChatMessage[data-testid="chat-message-assistant"] > div:nth-child(2) {{
        background: {ASSIST_BUBBLE} !important;
        color: {TEXT} !important;
        border-radius: {RADIUS};
        padding: 14px 16px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# === Load FAISS vectorstore ===
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# === Custom grounded prompt ===
custom_prompt = """
You are a helpful medical assistant chatbot.
Use ONLY the provided context to answer the user’s question.
If the context does not contain relevant info, say politely: 
"I don’t have enough medical information in my sources to answer that. Please consult a healthcare professional."
Do NOT make things up.
Always remind the user that you are not a doctor and this is not medical advice.

Previous discussion (summary):
{chat_summary}

Context:
{context}

Question:
{question}

Answer:
"""

def main():
    st.markdown(
        """
        <span style="font-size:28px;">🩺</span>
        <h1 style="margin:0;">Medical Chatbot NTI</h1>
        <hr/>
        """,
        unsafe_allow_html=True,
    )

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Replay history
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message['role'] == 'user' else "🤖"
        st.chat_message(message['role'], avatar=avatar).markdown(message['content'])

    prompt = st.chat_input("How can I help you today?")

    if prompt:
        # Handle rude input
        if any(word in prompt.lower() for word in ["stupid", "idiot", "useless"]):
            response_text = "I understand you’re frustrated. Let’s refocus—what medical issue can I help with?"
            st.chat_message('assistant', avatar="🤖").markdown(response_text)
            st.session_state.messages.append({'role': 'assistant', 'content': response_text})
            return

        st.chat_message('user', avatar="🧑‍💻").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            # === Setup LLM (Streaming Enabled) ===
            GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.4,
                max_tokens=512,
                streaming=True,  # STREAMING ENABLED
                api_key=GROQ_API_KEY,
            )

            # === Retriever ===
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.7}
            )

            # === Memory (modern version with explicit input keys) ===
            if 'memory' not in st.session_state:
                buffer_memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    input_key="question"
                )
                summary_memory = ConversationSummaryBufferMemory(
                    llm=llm,
                    memory_key="chat_summary",
                    return_messages=False,
                    input_key="question",
                    max_token_limit=1000
                )
                st.session_state.memory = CombinedMemory(memories=[buffer_memory, summary_memory])

            # === Prompt template ===
            prompt_template = PromptTemplate(
                template=custom_prompt,
                input_variables=["context", "question", "chat_summary"]
            )

            # === Build conversational RAG chain ===
            rag_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.memory,
                combine_docs_chain_kwargs={"prompt": prompt_template}
            )

            # === Streamed response (filtered for text only) ===
            with st.chat_message("assistant", avatar="🤖"):
                placeholder = st.empty()
                full_response = ""

                for chunk in rag_chain.stream({'question': prompt}):
                    if "answer" in chunk:
                        text_part = chunk["answer"]
                        full_response += text_part
                        placeholder.markdown(full_response)

                placeholder.markdown(full_response)

            # Store the final response in session
            st.session_state.messages.append({'role': 'assistant', 'content': full_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
