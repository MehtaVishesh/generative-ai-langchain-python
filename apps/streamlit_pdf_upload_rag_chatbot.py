"""Streamlit chatbot for asking questions about uploaded PDF files.

The app loads uploaded PDFs, splits their text into chunks, indexes the chunks
for semantic search, and gives an agent a retrieval tool to answer questions
from that content. The vector store, agent checkpoint, and chat history are
kept in memory for the active Streamlit session; they are not durable storage.
"""

# Import the components used to build the PDF retrieval and chat workflow.
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool 
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

# Read API credentials from the local environment (for example, a .env file).
load_dotenv()

# Track whether the current session has processed uploaded PDFs and built an agent.
if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Build the searchable PDF index and the agent that will query it.
def process_documents(path):
    # Load every PDF saved in the upload directory, then preserve page metadata
    # while splitting the extracted text into overlapping retrieval chunks.
    pdf_loader = PyPDFDirectoryLoader(path)
    loaded_pdf_documents = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunked_pdf_documents = text_splitter.split_documents(loaded_pdf_documents)

    # Embed each chunk and keep its vector in an in-memory semantic index.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore.from_documents(
        documents=chunked_pdf_documents,
        embedding=embeddings
    )

    # The agent can request relevant passages through this narrowly scoped tool.
    llm = ChatGroq(model = "openai/gpt-oss-20b")

    @tool
    def retrieve_document_context(user_query: str) -> str:
        """Return the three most relevant passages from the uploaded PDFs."""
        matching_documents = vector_store.similarity_search(user_query, k=3)
        retrieved_context = ""
        for match in matching_documents:
            retrieved_context += match.page_content
        return retrieved_context

    # Let the model decide when to retrieve, while instructing it to ground
    # answers in the uploaded documents and acknowledge missing information.
    agent = create_agent(
        model=llm,
        tools=[retrieve_document_context],
        system_prompt="""
        Use "retrieve_document_context" to answer questions only 
        using data from the PDFs. If you cannot find the answer, 
        say "I don't know".
        """,
        checkpointer= InMemorySaver()
    )

    # Keep the agent (and its in-memory vector store) available across reruns
    # in this browser session, and switch the UI from upload to chat mode.
    st.session_state.agent = agent
    st.session_state.documents_uploaded = True

# Collect multiple PDFs, save them locally for the directory loader, and
# process them before rerunning into the question-and-answer interface.
if not st.session_state.documents_uploaded:
    uploaded_files = st.file_uploader(
        label = "Select PDF files",
        type = ["PDF"],
        accept_multiple_files = True
        )
    if uploaded_files:
        with st.spinner("Processing..."):
            path = "./pdf_files/"
            for file in uploaded_files:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())

            process_documents(path)
            st.rerun()

# Render the chat interface only after document processing has created an agent.
if st.session_state.documents_uploaded and st.session_state.agent:
    # Replay the session transcript on each rerun so prior turns remain visible.
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask anything related to PDFs")
    if query:
        # Save and display the question, then invoke the agent using a stable
        # thread ID so its in-memory checkpoint can continue this conversation.
        st.session_state.messages.append({
            "role":"user",
            "content":query
        })
        st.chat_message("user").markdown(query)
        res = st.session_state.agent.invoke(
            {"messages":[{"role":"user", "content":query}]},
            {"configurable": {"thread_id":1}}
        )

        # Store and display the agent's final reply for the next Streamlit rerun.
        response_content = res["messages"][-1].content
        st.session_state.messages.append({
                    "role":"ai",
                    "content":response_content
        })
        st.chat_message("ai").markdown(response_content)