import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load env variables (Ensure your .env has GROQ_API_KEY)
load_dotenv()

## Arxiv Tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

## Wikipedia Tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

## DuckDuck Search Tool
search = DuckDuckGoSearchRun(name="Search")

st.title("🔎 LangChain - Chat with Search")

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
        st.stop()

    # 1. Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", streaming=True)
    tools = [search, arxiv, wiki]

    # 2. Define the Prompt (REQUIRED for tool calling agents)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 3. Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt_template)

    # 4. Create the Executor (This handles the running and tool calling logic)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        # 5. Invoke the executor
        response = agent_executor.invoke(
            {"input": prompt}, 
            callbacks=[st_cb]
        )
        
        output = response["output"]
        st.session_state.messages.append({'role': 'assistant', "content": output})
        st.write(output)