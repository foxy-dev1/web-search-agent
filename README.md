
# ChatBot with Tool Integration

A Python-based chatbot built with LangChain and LangGraph, integrating multiple tools such as Wikipedia search, web search, and Gmail draft creation. This project leverages Google's Gemini model for natural language processing and maintains chat history using SQLite. The chatbot is deployed using Streamlit for an interactive user interface.

## Features

- **Tool Integration**: 
  - `wikipedia_search`: Fetches summaries from Wikipedia for general knowledge queries.
  - `web_search_tool`: Searches the web using Searx for real-time or latest information.
  - `gmail_tool`: Creates email drafts in Gmail using the Gmail API.
- **Dynamic Tool Selection**: Automatically selects the appropriate tool based on the user's query (e.g., historical facts vs. current events).
- **Chat History**: Stores conversation history in an SQLite database for context-aware responses.
- **Web Scraping**: Uses Playwright and Selenium for screenshot-based OCR and content extraction from web pages.
- **Streamlit UI**: Provides a simple chat interface for user interaction.
- **Customizable**: Modular design allows easy addition of new tools or features.

## Prerequisites

- Python 3.8+
- Google API Key (for Gemini model)
- Gmail API credentials (for Gmail tool)
- SQLite (included with Python)
- A running Searx instance (for web search)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/foxy-dev1/web-search-agent.git
   cd web-search-agent
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
 

3. **Set Up Environment Variables**:
   Create a `.env` file in the root directory and add your Google API key:
   ```
   GOOGLE_API_KEY=<your-google-api-key>
   ```

4. **Configure Gmail API**:
   - Download your Gmail API `credentials.json` from the Google Cloud Console.
   - Place it in the project root directory.
   - Run the script once to generate a `token.json` file for authentication.

5. **Set Up Searx**:
   - Run a local Searx instance (e.g., via Docker):
     ```bash
     mkdir my-instance
     cd my-instance
     export PORT=8080
     docker pull searxng/searxng
     docker run --rm -d -p ${PORT}:8080 -v "${PWD}/searxng:/etc/searxng" -e "BASE_URL=http://localhost:$PORT/" -e "INSTANCE_NAME=my-instance" searxng/searxng

## Usage

1. **Run the Application**:
   ```bash
   streamlit run app_new.py
   ```

2. **Interact with the Chatbot**:
   - Open your browser at `http://localhost:8501`.
   - Enter a query in the text input field (e.g., "Tell me about Albert Einstein", "What's the latest news on AI?", or "Draft an email to example@gmail.com").
   - The chatbot will respond with answers fetched using the appropriate tool.

3. **Example Queries**:
   - "Who was Cleopatra?" → Uses `wikipedia_search`.
   - "What’s the latest score of the Lakers?" → Uses `web_search_tool`.
   - "Draft an email to friend@gmail.com with subject 'Hello'" → Uses `gmail_tool`.

## Project Structure

```
web-search-agent/
│
├── app_new.py      # Main script with chatbot logic
├── credentials.json      # Gmail API credentials (not tracked in git)
├── token.json           # Gmail API token (generated after auth)
├── screenshot.png       # Temporary file for web screenshots
├── new_chat_memory.db   # SQLite DB for chat history
├── display_memory.db    # SQLite DB for display history
├── .env                 # Environment variables
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## How It Works

1. **State Management**: Uses `langgraph` to maintain a stateful conversation flow.
2. **Tool Execution**: Dynamically invokes tools based on user input and LLM tool calls.
3. **Database**: Stores chat history and display messages in SQLite for persistence.
4. **UI**: Streamlit renders the chat interface and displays tool usage details.

## Limitations

- Requires a local Searx instance for web searches (or a public instance).
- Gmail tool needs pre-configured API credentials.
- OCR functionality depends on Tesseract and may fail on complex web pages.
- No built-in error recovery for failed tool executions.

