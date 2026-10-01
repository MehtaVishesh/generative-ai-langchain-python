# Streamlit renders the chat UI; LangChain sends the conversation to the model.
from langchain_openai import ChatOpenAI
import streamlit as st

# Configure the chat model used to answer each question.
llm = ChatOpenAI(
    model="gpt-6-luna",
)

# Streamlit reruns this script after interactions. Keep the transcript in
# session_state so it survives those reruns for the active browser session.
# This is in-memory session history, not durable storage across restarts.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show the app heading and collect the next question.
st.title("Meet Vish")
st.markdown("A QnA Chatbot using LangChain and GPT6-Luna")
query = st.chat_input("The only limit is your imagination. Ask anything")

# Rebuild the visible conversation from the transcript on every rerun.
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

if query:
    # Add the question before invoking the model so it receives the new turn
    # together with earlier turns in this session.
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role":"user", "content":query})
    res = llm.invoke(st.session_state.messages)

    # Show and store the reply so it appears again on the next rerun.
    st.chat_message("ai").markdown(res.content)
    st.session_state.messages.append({"role":"ai", "content":res.content})

