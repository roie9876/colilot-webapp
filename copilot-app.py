import streamlit as st
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import re
import json

st.set_page_config(layout="wide")

load_dotenv()

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

DEFAULT_SYSTEM_MESSAGE = (
    "You are a highly knowledgeable and precise coding assistant. "
    "Always format your responses using Markdown. Clearly explain your reasoning, provide code blocks enclosed "
    "in triple backticks (` ``` `), and specify the language (e.g., `cpp`). "
    "Include comprehensive inline comments and explanations within your code."
)

if "system_message" not in st.session_state:
    st.session_state.system_message = DEFAULT_SYSTEM_MESSAGE

with st.expander("Edit System Message"):
    st.session_state.system_message = st.text_area("System Message", st.session_state.system_message, height=150)

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