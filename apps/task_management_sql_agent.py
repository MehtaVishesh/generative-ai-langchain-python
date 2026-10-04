"""Task management agent powered by a SQLite database.

This Streamlit app lets a user manage tasks through a SQL-capable agent.
The interface is intentionally simple and readable so that recruiters and
learners can follow the flow of the application without needing to decipher
complex logic or hidden state management.
"""

# Load API credentials and environment settings from the local .env file.
from dotenv import load_dotenv
load_dotenv()

# Streamlit provides the chat interface; LangChain and LangGraph provide the
# model, SQL tooling, and agent orchestration used to manage the database.
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit as st
from uuid import uuid4

# Connect to the SQLite database and create the tasks table if it does not yet
# exist. This keeps the app ready for task creation, updates, and retrieval.
db = SQLDatabase.from_uri("sqlite:///my_tasks.db")
db.run("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task TEXT NOT NULL,
        description TEXT NOT NULL,
        status TEXT CHECK(status IN ('pending', 'in_progress', 'completed')) NOT NULL DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# Configure the language model and attach the SQL tooling that exposes
# database actions to the agent.
llm = ChatGroq(model="openai/gpt-oss-20b")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
system_prompt = """
You are a task management assitant that interacts with a SQL database containing a "tasks" table.
TASK RULES:
1. Limit SELECT queries to 10 results maximum with ORDER BY created_at DESC.
2. After CREATE, UPDATE, or DELETE operations, confirm with SELECT Queries to show the current state of the tasks table.
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser.
CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(task, description, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 10
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR task=?
    DELETE: DELETE FROM tasks WHERE id=? OR task=?
Table schema: id, task, description, status(pending/in_progress/completed), created_at
"""

# The following block is intentionally left as a comment for debugging and
# inspection during development.
# for tool in tools:
#     print(tool.name)

# Streamlit reruns this script after interactions. Keep a stable thread ID in
# session_state so the agent's checkpointer can resume this conversation.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

# Cache the agent as a shared resource across reruns; its checkpointer is
# created with the agent and uses thread_id to keep graph state per conversation.
@st.cache_resource
def get_agent():
    return create_agent(
        model=llm,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt
    )
agent = get_agent()

# Render the page heading and provide a clear entry point for task-related
# prompts.
st.subheader("Task Management Agent with SQL Database")
query = st.chat_input("Ask me anything about your tasks, and I will manage them for you")

# Keep a separate per-session transcript for rendering chat turns after reruns;
# the agent checkpointer stores graph state, not this displayed history.
if "history" not in st.session_state:
    st.session_state.history = []

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

# A submitted prompt triggers a rerun. Only process the user query when it is
# nonempty, then display the result in the chat panel.
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":query})
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            res = agent.invoke(
                {"messages":[{"role":"user","content":query}]},
                {"configurable":{"thread_id":st.session_state.thread_id}}
            )
            result = res["messages"][-1].content
            st.markdown(result)
            st.session_state.history.append({"role":"ai","content":result})
