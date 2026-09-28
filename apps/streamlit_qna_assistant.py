from langchain_openai import ChatOpenAI
import streamlit as st

llm = ChatOpenAI(
    model="gpt-6-luna",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Meet Vish")
st.markdown("A QnA Chatbot using LangChain and GPT6-Luna")
query = st.chat_input("The only limit is your imagination. Ask anything")

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role":"user", "content":query})
    res = llm.invoke(st.session_state.messages)
    st.chat_message("ai").markdown(res.content)
    st.session_state.messages.append({"role":"ai", "content":res.content})

