# from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, trim_messages
from langchain_core.tools import tool, ToolException, InjectedToolArg
# from langchain_core.runnables import RunnableConfig
# from langchain_community.utilities import ArxivAPIWrapper
# from langchain_community.tools import ArxivQueryRun, HumanInputRun
from langgraph.graph import StateGraph,START,END, add_messages, MessagesState
# from langgraph.prebuilt import create_react_agent, ToolNode
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.store.base import BaseStore
# from langgraph.store.memory import InMemoryStore
from typing import Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import wikipedia
# import uuids
import operator
# from IPython.display import Image, display
import os

# import trafilatura
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_community.utilities import SearxSearchWrapper

from selenium import webdriver
from selenium.webdriver.common.by import By

import streamlit as st
from langchain_google_community import GmailToolkit

from langchain_google_community.gmail.utils import (build_resource_service,get_gmail_credentials)

import sqlite3
import json

from playwright.sync_api import sync_playwright
import time
import cv2
import pytesseract

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


# load the model 
gemini_api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                               api_key=gemini_api_key)


# using playwright to get the screenshot of the web page

def access_page_tool(url:str):

    """takes a url as input takes the screenshot by acessing the browser and return ocr test from it """

    with sync_playwright() as playwright:
            webkit = playwright.webkit
            browser = webkit.launch()
            context = browser.new_context()
            page = context.new_page()
            page.goto(url)
            # if page.locator("text=OK").is_visible():
            #     page.locator("text=OK").click()
            time.sleep(2)
            page.on("dialog", lambda dialog: dialog.dismiss())  

            time.sleep(6)

            page.screenshot(path="screenshot.png")
            browser.close()


    print("accessing the image")
    img = cv2.imread('./screenshot.png')

    print("py ocr to get the text")

    res = pytesseract.image_to_string(img)
    if res:
        print("text extracted success")


    return res


# search logic and search tool

def search_web(query: str):
    """Fetch search results using Searx API."""
    search = SearxSearchWrapper(searx_host="http://localhost:8080")

    try:
        results = search.results(query=query, num_results=2, time_range="year")

        if results:
            print(" Success: Loaded search results")
            return results
        else:
            print("Warning: No results found")
            return []
    
    except Exception as e:
        print(f" Error: Failed to fetch results from Searx - {e}")
        return []

def get_title_content(results: list):
    """Extract content from the fetched search results."""
    res = []

    for result in results:
        url = result.get("link")
        title = result.get("title", "Untitled")

        if not url:
            continue  

        try:
            # response = trafilatura.fetch_url(url)
            # content = trafilatura.extract(response) if response else None

            options = webdriver.ChromeOptions()
            options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")
            options.add_argument("headless")

            driver = webdriver.Chrome(options=options)
            driver.get(url)

            elements = driver.find_elements(By.XPATH, "/html/body")

            for ele in elements:
                content = ele.text

            if content:
                res.append({"title": title, "content": content})

            else:
                response_text = access_page_tool(url)

                if response_text:
                    res.append(response_text) 

                else:
                    print(f" Warning: No content extracted from {url}")

        except Exception as e:
            print(f" Error: Failed to extract content from {url} - {e}")

    if res:
        print(" Success: Loaded extracted content")

    return res



class SearchQuery(BaseModel):
    query: str = Field(description="Query to be used to search the web")


# websearch tool

@tool(args_schema=SearchQuery)
def web_search_tool(query: str) -> list:
    """This tool searches the web and returns a list with the title and content of web pages."""
    
    results = search_web(query=query)

    if not results:
        return [{"error": "No search results found."}]

    res = get_title_content(results)

    if not res:
        return [{"error": "No content could be extracted from the search results."}]

    return res






# wikipedia search tool

class wikipedia_topic(BaseModel):

    topic:str = Field(description="topic to be searched using wikipedia tool ")



@tool (args_schema=wikipedia_topic)
def wikipedia_search(topic:str)->str:
    
    """tool to search wikipedia using the provided topic returns a summary of the topic"""
    summary = wikipedia.summary(topic,auto_suggest=False)

    return summary





# gmail draft making tool 

def intialize_gmail_auth():
    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=['https://www.googleapis.com/auth/gmail.compose'],
        client_secrets_file="./credentials.json",
    )

    api_resource = build_resource_service(credentials=credentials)

    toolkit = GmailToolkit(api_resource=api_resource)

    print("auth with gmail completed")

    return toolkit


class gmail_args(BaseModel):

    message:str = Field(description="message to be included in the draft")
    to:list[str] = Field(description="email address to send to")
    subject:str = Field(description="subject of the draft mail") 


@tool (args_schema=gmail_args)

def gmail_tool(message:str, to:list[str], subject:str):

    """ tool to interact with the gmail api and create a draft email"""
    try:
        toolkit = intialize_gmail_auth()

        tools = toolkit.get_tools()

        create_draft_tool = next(
            (tool for tool in tools if tool.name == "create_gmail_draft"),
            None
        )
        
        if create_draft_tool is None:
            raise ValueError("create_gmail_draft tool not found in toolkit")
        
        draft_input = {
            "message": message,
            "to": to,
            "subject": subject
        }
        
        result = create_draft_tool.invoke(draft_input)
        return f"Draft created successfully: {result}"
    
    except Exception as e:
        return f"Error creating draft: {str(e)}"





# state initialization and tool exists logic

class State(TypedDict):

    messages: Annotated[list[AnyMessage],operator.add]


tools = [web_search_tool,wikipedia_search,gmail_tool]

tools_name = {t.name: t for t in tools}
model = model.bind_tools(tools)

def tool_exists(state:State):

    result = state['messages'][-1].tool_calls

    return len(result)>0;





# tool execute logic to call tool and perform tasks

def tool_execute(state: State):
    results = []
    tool_calls = state['messages'][-1].tool_calls

    for t in tool_calls:
        if t['name'] not in tools_name:
            results.append(
                ToolMessage(tool_call_id=t['id'], name=t['name'], content="Tool not found. Try again.")
            )
            continue
        
        try:
            tool = tools_name[t['name']]

            result = tool.run(t['args'])
            
            results.append(
                ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result))
            )
            
        except Exception as e:
            error_msg = f"Error executing tool: {str(e)}"

            results.append(
                ToolMessage(tool_call_id=t['id'], name=t['name'], content=error_msg)
            )

    return {"messages": results}



# run llm logic 


def run_llm(state:State):

    messages = state['messages']
    
    message = model.invoke(messages)
    
    return {'messages': [message]}






# graph initialization with nodes and edges


graph = StateGraph(State)
graph.add_node("llm",run_llm)
graph.add_node("tools",tool_execute)

graph.add_conditional_edges("llm",tool_exists,{True:"tools",False:END})
graph.set_entry_point("llm")

graph.add_edge("tools","llm")


graph = graph.compile()


system_message = SystemMessage(content="""
You are a helpful assistant designed to answer user questions accurately by using the available tools.  
You have access to **two tools**:  
1. **wikipedia_search** – Use this tool to retrieve general knowledge or well-documented topics.  
2. **web_search_tool** – Use this tool to fetch real-time or the latest information from the web.  
3. if user asks to know the latest score or next match of a team use the web_search_tool to find the relevant information
   and only provide whats asked by the user. 
4. when asked about a famous person or celebrity use wikipedia_search tool if the information is not found on the wikipedia_tool use the web_search_tool for more information
5. if user asks to draft a email to a particular gmail or email address use the gmail_tool to draft the email
6. when asked about the history or what did i asked you earlier please provide the output of it in neat format doesnt need to be same wording as the question asked earlier

### **How to Use the Tools:**  
- If the user asks about **historical facts, definitions, or general knowledge**, use `wikipedia_search`.  
- If the user asks about **current events, news, or real-time updates**, use `web_search_tool`.  
- If a query is unclear, choose the most appropriate tool to get relevant information.  

### **Your Task:**  
- Select the best tool based on the user's query.  
- Retrieve information using the tool.  
- Use the retrieved information to generate a **clear, concise, and complete answer** for the user.  
- If the tools do not return useful results, politely inform the user that you couldn't find relevant information.  

**You do not have any external knowledge beyond these tools. Always rely on them to answer user queries.**  
""")




# set up db to store chats

conn = sqlite3.connect("new_chat_memory.db")
cursor = conn.cursor()

cursor.execute("""

    CREATE TABLE IF NOT EXISTS chat_history(
               
               thread_id TEXT PRIMARY KEY,
               messages TEXT
               
               )
""")


def save_chat_history(thread_id,new_messages):

    cursor.execute("""

        INSERT INTO chat_history (thread_id,messages)
        VALUES (?,json(?))
        ON CONFLICT(thread_id)
        DO UPDATE SET messages = json_insert(messages,'$[#]', json(?))
        """,
        (thread_id,json.dumps(new_messages),json.dumps(new_messages)),

        )
    
    conn.commit()


config = {"configurable":{"thread_id":"1"}}


def flatten_messages(messages):

    flat_list = []
    for msg in messages:
        if isinstance(msg,list):
            flat_list.extend(flatten_messages(msg))

        else:
            flat_list.append(msg)

    return flat_list


# need to add coz to meet the langchain name format
def format_name(name):

    if name == "HumanMessage":
        name = 'human'
    
    elif name == "AIMessage":
        name = 'ai'
    
    elif name == 'SystemMessage':
        name = 'system'
    
    elif name == 'ToolMessage':
        name = 'tool'

    return name




# db to display user msg, tool used, ai msg 

conn_new = sqlite3.connect("display_memory.db")
cursor_new = conn_new.cursor()

cursor_new.execute("""

    CREATE TABLE IF NOT EXISTS display_history(
               
               thread_id TEXT PRIMARY KEY,
               messages TEXT
               
               )
""")


def save_display_history(thread_id,new_messages):

    cursor_new.execute("""

        INSERT INTO display_history (thread_id,messages)
        VALUES (?,json(?))
        ON CONFLICT(thread_id)
        DO UPDATE SET messages = json_insert(messages,'$[#]', json(?))
        """,
        (thread_id,json.dumps(new_messages),json.dumps(new_messages)),

        )
    
    conn_new.commit()


# config = {"configurable":{"thread_id":"1"}}

# to get tools used info

def get_tool_call_info(result):

    tool_call_info = []


    for msg in result['messages']:
        if isinstance(msg,AIMessage):

            tool_calls = msg.tool_calls

            if tool_calls != []:
                for args in tool_calls:
                    if args['name'] == 'wikipedia_search':
                        return {"name":args['name'],"topic":args['args']['topic']}
                    elif args['name'] == 'web_search_tool':
                        return {"name":args['name'],"topic":args['args']['query']}
                    else:
                        return {"name":args['name'],"topic":args['args']['subject']}


def get_stored_info_display_db():
    
    cursor_new.execute("SELECT messages FROM display_history ")
    rows = cursor_new.fetchone()  

    if rows:
        msg = json.loads(rows[0])

        flatten_msg = [ x for x in flatten_messages(msg) if x['role'] != 'tool' ]

        return flatten_msg
    else:
        return None





st.title("chat bot")

user_input = st.text_input("enter query")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# for displaying messages in a chat
if "display_message" not in st.session_state:
    st.session_state["display_message"] = []





def chat_bot():

    thread_id = config.get("configurable",{}).get("thread_id")

    if user_input:
        

        st.session_state["display_message"].append({"role":"user","content":user_input})

    cursor.execute("SELECT messages FROM chat_history ")
    rows = cursor.fetchone()  

    if not rows:
        msg = [system_message, HumanMessage(content=user_input)]
    else:
        
        messages = json.loads(rows[0])

        # tool removed as tool interfers with tool_calls function
        messages = [ x for x in flatten_messages(messages) if x['role'] != 'tool' ]

        st.session_state['messages'] = messages

        msg = [HumanMessage(content=user_input)]

    context = st.session_state["messages"]
    result = graph.invoke({"messages": context + msg},config=config)


    # store the tool info and user / assistant msg to db
    tool_call_info = get_tool_call_info(result)
    new_messages = []

    user_msg_format = {"role":"user","content":user_input}

    ai_msg_format = {"role":"assistant","content":result['messages'][-1].content,"tool_used":tool_call_info}

    new_messages.append(user_msg_format)
    new_messages.append(ai_msg_format)


    # session state data
    if result:
        st.session_state["display_message"].append({"role":"assistant","content":result['messages'][-1].content})


    # history to be saved for context building
    existing_history = []

    n = len(result['messages'])

    for i in range(n):
        res = result['messages'][i]

        # need to do this as we are adding from result so it adds a dict as role which causes error
        if isinstance(res,dict):
            continue

        content = str(res)

        history_format = {"role":format_name(type(res).__name__),"content":content}

        existing_history.append(history_format)

    save_chat_history(thread_id,existing_history)
    save_display_history(thread_id,new_messages)

    return result



if user_input:

    res = chat_bot()   

flatten_msg = get_stored_info_display_db()


if flatten_msg:

    for msg in flatten_msg:
        if msg['role'] == 'user':
            st.chat_message(msg['role']).markdown(msg['content'])
        
        elif msg['role'] == 'assistant' and msg['tool_used'] != None:

            if msg['tool_used']['name'] == 'wikipedia_search':

                st.chat_message("tool").markdown(f"Tool used  '{msg['tool_used']['name']}' query used '{msg['tool_used']['topic']}'")
                st.chat_message(msg["role"]).markdown(msg['content'])

            elif msg['tool_used']['name'] == 'web_search_tool':  

                st.chat_message("tool").markdown(f"Tool used  '{msg['tool_used']['name']}' query used '{msg['tool_used']['topic']}'")
                st.chat_message(msg["role"]).markdown(msg['content'])

            else:

                st.chat_message(msg["role"]).markdown(msg['content'])

        else:

            st.chat_message(msg["role"]).markdown(msg['content'])