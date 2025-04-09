import streamlit as st
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import re
import json
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
st.set_page_config(layout="wide")

load_dotenv()
MAX_INPUT_TOKENS = int(os.getenv("AZURE_OPENAI_MAX_INPUT_TOKENS", "200000"))
MAX_COMPLETION_TOKENS = int(os.getenv("AZURE_OPENAI_MAX_COMPLETION_TOKENS", "100000"))

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",  # Adjust based on your Azure API version
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

MODEL_DEPLOYMENTS = {
    "o1": os.getenv("AZURE_OPENAI_DEPLOYMENT_O1"),
    "o3": os.getenv("AZURE_OPENAI_DEPLOYMENT_O3"),
    "gpt-4.5": os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT45")
}

DEFAULT_SYSTEM_MESSAGE = """
You are an expert AI assistant specialized in analyzing and summarizing technical documents, especially academic papers and technical PDFs. Always structure your responses clearly using Markdown.

When analyzing PDFs:

- Clearly summarize the document’s key ideas, algorithms, methodologies, and conclusions.
- Highlight important mathematical concepts and accurately explain any formulas.
- Provide insightful commentary on the document’s practical applications, strengths, and potential weaknesses.
- Suggest specific improvements or clarifications if applicable.

When providing code or examples:

- Format all code in clearly labeled Markdown code blocks enclosed in triple backticks (` ``` `).
- Always specify the programming language explicitly (e.g., `python`, `cpp`, `matlab`).
- Include comprehensive inline comments and clear explanations of your logic.

Aim to deliver responses that are insightful, accurate, practical, and immediately helpful to the user's specific task.
"""

if "system_message" not in st.session_state:
    st.session_state.system_message = DEFAULT_SYSTEM_MESSAGE

with st.expander("Edit System Message"):
    st.session_state.system_message = st.text_area("System Message", st.session_state.system_message, height=150)

# --- Begin PDF Upload Section using Azure Document Intelligence ---
uploaded_pdf = st.file_uploader("Upload a PDF (with images)", type=["pdf"])
if uploaded_pdf is not None:
    try:
        # Read PDF file content bytes
        pdf_bytes = uploaded_pdf.getvalue()
        
        # Setup Azure Document Intelligence Client

        
        form_recognizer_endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
        form_recognizer_key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
        
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=form_recognizer_endpoint,
            credential=AzureKeyCredential(form_recognizer_key)
        )
        
        # Analyze the document using the prebuilt-layout model
        poller = document_intelligence_client.begin_analyze_document("prebuilt-layout", body=pdf_bytes)
        result = poller.result()
        
        extracted_text = ""
        # Extract text from each page's lines in order
        for page in result.pages:
            for line in page.lines:
                extracted_text += line.content + "\n"
                
            # Optionally: process tables or mathematical formulas here if needed
            
        st.markdown("### Extracted PDF Text")
        st.text_area("PDF Content", extracted_text, height=200)
        
        # Button to trigger PDF analysis via LLM
        if st.button("Analyze PDF"):
            if extracted_text.strip() == "":
                st.error("No text was extracted from the PDF. Please check your document or try another file.")
            else:
                pdf_message = "Below is the extracted text from the uploaded PDF for analysis:\n\n" + extracted_text
                st.session_state.messages.append({"role": "user", "content": pdf_message})
                st.success("PDF content added to the conversation for analysis.")
    except Exception as e:
        st.error("Failed to process PDF using Azure Document Intelligence: " + str(e))
# --- End PDF Upload Section ---

if st.button("Clear Chat"):
    st.session_state.messages = [{"role": "system", "content": st.session_state.system_message}]
    
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": st.session_state.system_message}]
else:
    # Keep the first message in sync with the updated system message
    st.session_state.messages[0]["content"] = st.session_state.system_message

model_choice = st.selectbox("Select Model", ["o1", "o3", "gpt-4.5"])

chat_container = st.container()

with chat_container:
    # Display previous conversation
    for message in st.session_state.messages[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

user_input = st.chat_input("Ask a coding question:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                stream = client.chat.completions.create(
                    model=MODEL_DEPLOYMENTS[model_choice],
                    messages=st.session_state.messages,
                    reasoning_effort="high",
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                    stream=True,
                )

                collected_chunks = []
                for chunk in stream:
                    # Check if chunk.choices exists before accessing
                    if not chunk.choices or not chunk.choices[0].delta:
                        continue
                    content = chunk.choices[0].delta.content or ""
                    collected_chunks.append(content)

                full_response = "".join(collected_chunks)
                message_placeholder = st.empty()

                message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

with st.sidebar:
    st.header("Chat Sessions")

    # 1) Load existing sessions into a dictionary (if file exists)
    sessions_file = "chat_sessions.json"
    if os.path.exists(sessions_file):
        with open(sessions_file, "r") as f:
            all_sessions = json.load(f)
    else:
        all_sessions = {}

    # 2) Show a list of saved sessions in a selectbox
    saved_session_names = list(all_sessions.keys())
    selected_session = st.selectbox("Select Saved Session", [""] + saved_session_names)

    # 3) Button to load the selected existing session
    if st.button("Load Selected"):
        if selected_session:
            st.session_state.messages = all_sessions[selected_session].copy()

    st.write("---")
    session_name_input = st.text_input("Session Name to Save")
    # 4) Button to save the current session under a user-provided name
    if st.button("Save Current Session"):
        if session_name_input.strip():
            all_sessions[session_name_input] = st.session_state.messages
            with open(sessions_file, "w") as f:
                json.dump(all_sessions, f)
            st.success(f"Session saved as '{session_name_input}'")