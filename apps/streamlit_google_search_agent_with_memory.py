# Load API credentials and other settings from the local .env file.
from dotenv import load_dotenv
from uuid import uuid4
load_dotenv()

# Streamlit provides the chat interface; LangChain and LangGraph provide the
# model, search tool, agent orchestration, and conversation checkpointing.
import streamlit as st
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.checkpoint.memory import MemorySaver

# Configure the language model and the Google Serper search wrapper.
llm = ChatGroq(model="openai/gpt-oss-20b", streaming=True)
search = GoogleSerperAPIWrapper()

# Streamlit reruns this script from top to bottom after user interactions.
# Ordinary variables are recreated on each run, so session_state keeps the
# checkpointer and displayed chat history available between prompts.
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

# LangGraph uses a thread ID to associate turns with one conversation. Keep a
# unique ID in session_state so it survives reruns and remains stable per session.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

tools = [search.run]

# Build the agent with search access and a checkpointer so its graph state can
# be resumed for the same thread on later prompts.
agent = create_agent(
    model = llm,
    tools = tools,
    checkpointer = st.session_state.memory,
    system_prompt="If you can't answer with your knowledge base, lookup on Google, but never fail to answer"
)

# Render the app title and replay prior turns from Streamlit's session history.
st.subheader("I am Lightning - a Fast Google Search Agent with Memory")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

# A submitted prompt triggers a rerun; only process it when the input is nonempty.
query = st.chat_input("Ask me anything, and I will find the answer for you")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":query})    

    # Stream message chunks from the agent and pass the thread ID so the
    # checkpointer can load the matching conversation state.
    res = agent.stream(
        {"messages":[{"role":"user","content":query}]},
        {"configurable":{"thread_id":st.session_state.thread_id}},
        stream_mode = "messages"
    )

    # Keep one assistant bubble open and update its placeholder as chunks arrive.
    # This makes the answer appear progressively instead of waiting for completion.
    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()
        message = ""
        for chunk in res:
            message += chunk[0].content
            space.write(message)
        st.session_state.history.append({"role":"ai","content":message})
