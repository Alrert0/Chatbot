import streamlit as st
import hashlib
import sqlite3
import os
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_ollama import OllamaLLM
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms.ollama import Ollama
import matplotlib.pyplot as plt
import json
from datetime import datetime
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler
import asyncio
import nest_asyncio
import logging
import threading
import sys


try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    print("nest_asyncio module not found. Some features may not work as expected.")
try:
    from wordcloud import WordCloud
    wordcloud_available = True
except ImportError:
    wordcloud_available = False
    print("WordCloud module not found. Word cloud visualization will not be available.")

nest_asyncio.apply()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants and Initialization ---
llm_model = "llama3"
chroma_db_path = os.path.join(os.getcwd(), "chroma_db")
collection_name = "knowledge_base"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize ChromaDB
client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_or_create_collection(name=collection_name, metadata={"description": "Knowledge base for RAG"})

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Constants
SERPAPI_KEY = "227e6513442cfaa35c62f17d43e8f0de50c3450af033c0ed8f9e7bee4de393c2"
MAX_SEARCH_RESULTS = 5
CHAT_HISTORY_FILE = "chat_history.json"
TELEGRAM_BOT_TOKEN = "7890748687:AAFkzUuRXSpcW7SJtXD4o92s-_yJwR3sBdk"

# --- Functions ---
def generate_embeddings(documents):
    return embedding_model.encode(documents)

def save_embeddings(documents, ids):
    embeddings = generate_embeddings(documents)
    collection.add(documents=documents, ids=ids, embeddings=embeddings)

def query_chromadb(query_text, n_results=1):
    query_embedding = generate_embeddings([query_text])
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    return results["documents"][0] if results["documents"] else "No relevant documents found."

def google_search(query, num_results=MAX_SEARCH_RESULTS):
    if not SERPAPI_KEY:
        return "SerpAPI key not configured!"
    
    url = f"https://serpapi.com/search.json?q={query}&num={num_results}&api_key={SERPAPI_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        results = response.json().get("organic_results", [])
        if results:
            return "\n\n".join(f"Source: {item.get('link', 'Unknown')}\n{item.get('snippet', '')}" 
                             for item in results)
        return ""
    except requests.exceptions.RequestException as e:
        return f"Error accessing SerpAPI: {str(e)}"
    except json.JSONDecodeError:
        return "Error parsing SerpAPI response"

def wrap_text(text, max_length=80):
    return "\n".join([text[i:i + max_length] for i in range(0, len(text), max_length)])

def extract_text_from_file(file):
    if isinstance(file, str):  # For Telegram: file is a file path
        if file.endswith('.pdf'):
            loader = PyPDFLoader(file)
            documents = loader.load()
            return [doc.page_content for doc in documents]
        elif file.endswith('.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                return [f.read()]
    else:  # For Streamlit: file is a UploadedFile object
        if file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                loader = PyPDFLoader(temp_file.name)
                documents = loader.load()
            os.unlink(temp_file.name)
            return [doc.page_content for doc in documents]
        elif file.type == "text/plain":
            return [file.read().decode("utf-8")]
    
    return ["Unsupported file type."]

def process_query(user_input, num_chroma_results=3, num_google_results=5):
    chroma_context = query_chromadb(user_input, n_results=num_chroma_results)
    
    if chroma_context == "No relevant documents found.":
        context = google_search(user_input, num_results=num_google_results)
        source = "web search"
    else:
        context = chroma_context
        source = "knowledge base"

    prompt = f"""
    Question: {user_input}
    
    Context from {source}:
    {context}
    
    Please provide a comprehensive answer based on the context above.
    If using web search results, cite the sources.
    
    Answer:
    """

    try:
        llm = OllamaLLM(model=llm_model, base_url="http://localhost:11434")
        response = llm.invoke(prompt)
        return response, source
    except Exception as e:
        return f"Error generating response: {str(e)}", "error"

# --- Authentication Functions ---
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    conn.commit()
    conn.close()

def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username=?', (username,))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0] == make_hash(password)
    return False

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users VALUES (?,?)', (username, make_hash(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def save_chat_history(username, messages):
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        else:
            history = {}
        
        history[username] = {
            'messages': messages,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(CHAT_HISTORY_FILE, 'w') as f:
            json.dump(history, f)
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")

def load_chat_history(username):
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r') as f:
                history = json.load(f)
                return history.get(username, {}).get('messages', [])
        return []
    except Exception as e:
        logger.error(f"Error loading chat history: {str(e)}")
        return []

def create_wordcloud(text):
    if not wordcloud_available:
        logger.warning("WordCloud module not available. Skipping word cloud creation.")
        return None

    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception as e:
        logger.error(f"Error creating wordcloud: {str(e)}")
        return None

# --- Telegram Bot Functions ---
async def start(update: Update, context):
    keyboard = [
        [InlineKeyboardButton("🔑 Login", callback_data='login')],
        [InlineKeyboardButton("📝 Register", callback_data='register')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('🏛️ Welcome to Knowledge Navigator! Please login or register to continue:', reply_markup=reply_markup)

async def login(update: Update, context):
    await update.callback_query.answer()
    await update.callback_query.edit_message_text("🔤 Please enter your username and password in this format: login username password\nFor example: login Kayrat 123456")

async def register(update: Update, context):
    await update.callback_query.answer()
    await update.callback_query.edit_message_text("🔤 Please enter your desired username and password in this format: register username password\nFor example: register Maxim 123456")

async def process_login_register(update: Update, context):
    message_parts = update.message.text.split()
    if len(message_parts) != 3:
        await update.message.reply_text("❌ Invalid format. Please try again.")
        return

    action, username, password = message_parts

    if action.lower() == 'login':
        if check_user(username, password):
            context.user_data['logged_in'] = True
            context.user_data['username'] = username
            await send_main_menu(update, context)
        else:
            await update.message.reply_text("Invalid username or password. Please try again.")
    elif action.lower() == 'register':
        if add_user(username, password):
            await update.message.reply_text("✅ Registration successful! Please login.")
        else:
            await update.message.reply_text("Username already exists. Please choose a different username.")

async def send_main_menu(update: Update, context):
    keyboard = [
        [InlineKeyboardButton("💬 Ask a Question", callback_data='ask_question')],
        [InlineKeyboardButton("📚 Upload Document", callback_data='upload_document')],
        [InlineKeyboardButton("📜 View Chat History", callback_data='view_history')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(f"Welcome, {context.user_data['username']}! What would you like to do?", reply_markup=reply_markup)

async def button_click(update: Update, context):
    query = update.callback_query
    await query.answer()

    if query.data == 'login':
        await login(update, context)
    elif query.data == 'register':
        await register(update, context)
    elif query.data == 'ask_question':
        keyboard = [[InlineKeyboardButton("⏪ Back", callback_data='main_menu')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text="Please type your question.", reply_markup=reply_markup)
    elif query.data == 'upload_document':
        keyboard = [[InlineKeyboardButton("⏪ Back", callback_data='main_menu')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text="Please upload a PDF or TXT file.", reply_markup=reply_markup)
    elif query.data == 'view_history':
        username = context.user_data.get('username')
        if username:
            history = load_chat_history(username)
            if history:
                history_text = "\n\n".join([f"Q: {msg.get('content', 'No question')}\nA: {msg.get('response', 'No response')}" for msg in history[-5:]])
                keyboard = [[InlineKeyboardButton("⏪ Back", callback_data='main_menu')]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(text=f"Here are your last 5 interactions:\n\n{history_text}", reply_markup=reply_markup)
            else:
                keyboard = [[InlineKeyboardButton("⏪ Back", callback_data='main_menu')]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(text="No chat history found.", reply_markup=reply_markup)
        else:
            keyboard = [[InlineKeyboardButton("⏪ Back", callback_data='main_menu')]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(text="Please login first to view chat history.", reply_markup=reply_markup)
    elif query.data == 'main_menu':
        await send_main_menu(update, context)

async def send_main_menu(update: Update, context):
    keyboard = [
        [InlineKeyboardButton("💬 Ask a Question", callback_data='ask_question')],
        [InlineKeyboardButton("📚 Upload Document", callback_data='upload_document')],
        [InlineKeyboardButton("📜 View Chat History", callback_data='view_history')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    message_text = f"🖐️ Welcome, {context.user_data['username']}! What would you like to do?"
    
    if isinstance(update, Update) and update.callback_query:
        await update.callback_query.edit_message_text(message_text, reply_markup=reply_markup)
    elif isinstance(update, Update) and update.message:
        await update.message.reply_text(message_text, reply_markup=reply_markup)
    else:
        logger.warning("Unexpected update type in send_main_menu")

async def handle_message(update: Update, context):
    if not context.user_data.get('logged_in'):
        if update.message.text.lower().startswith(('login', 'register')):
            await process_login_register(update, context)
        else:
            await update.message.reply_text("Please login or register first.")
        return

    user_input = update.message.text
    username = context.user_data['username']

    await update.message.reply_text("🔜 Processing your query...")
    
    response, source = process_query(user_input)
    
    await update.message.reply_text(f"Answer:\n{response}\n\nSource: {source}")
    
    messages = load_chat_history(username)
    messages.append({"content": user_input, "response": response, "source": source})
    save_chat_history(username, messages)

    keyboard = [
    [InlineKeyboardButton("💬 Ask Another Question", callback_data='ask_question')],
    [InlineKeyboardButton("📚 Upload Document", callback_data='upload_document')],
    [InlineKeyboardButton("📜 View Chat History", callback_data='view_history')],
]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('🔍 What would you like to do next?', reply_markup=reply_markup)

async def handle_document(update: Update, context):
    file = await context.bot.get_file(update.message.document.file_id)
    file_name = update.message.document.file_name
    file_path = os.path.join(tempfile.gettempdir(), file_name)
    await file.download_to_drive(file_path)
    
    texts = extract_text_from_file(file_path)
    if texts[0] != "Unsupported file type.":
        for i, content in enumerate(texts):
            new_id = f"doc_{collection.count() + 1}_{i}"
            save_embeddings([content], [new_id])
        await update.message.reply_text("✅ Document processed and added to knowledge base.")
    else:
        await update.message.reply_text("❌ Sorry, I can only process PDF and TXT files.")
    
    os.remove(file_path)

    keyboard = [
        [InlineKeyboardButton("💬 Ask a Question", callback_data='ask_question')],
        [InlineKeyboardButton("📚 Upload Another Document", callback_data='upload_document')],
        [InlineKeyboardButton("📜 View Chat History", callback_data='view_history')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text('🔍 What would you like to do next?', reply_markup=reply_markup)

# --- Streamlit App ---
def streamlit_app():
    st.set_page_config(page_title="🏛️ Knowledge Navigator", layout="wide")

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ''

    if not st.session_state.logged_in:
        st.title("🏛️ Welcome to Knowledge Navigator")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.header("Login")
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
                if check_user(login_username, login_password):
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        with tab2:
            st.header("Register")
            reg_username = st.text_input("Username", key="reg_username")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_password_confirm = st.text_input("Confirm Password", type="password")
            
            if st.button("Register"):
                if reg_password != reg_password_confirm:
                    st.error("❌ Passwords do not match.")
                elif len(reg_password) < 6:
                    st.error("❌ Password must be at least 6 characters long.")
                elif add_user(reg_username, reg_password):
                    st.success("✅ Registration successful! Please login.")
                else:
                    st.error("❌ Username already exists")

    else:
        st.title(f"🏛️ Welcome back, {st.session_state.username}!")
        
        with st.sidebar:
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = ''
                st.rerun()
            
            st.divider()
            
            if st.button("Clear Chat History"):
                save_chat_history(st.session_state.username, [])
                st.rerun()

        tab1, tab2, tab3 = st.tabs(["Chat", "Documents", "Analytics"])
        
        with tab1:
            with st.expander("💬 Chat with the AI", expanded=True):
                if 'messages' not in st.session_state:
                    st.session_state.messages = load_chat_history(st.session_state.username)

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        if "source" in message:
                            st.caption(f"Source: {message['source']}")

                user_input = st.text_input("🔍 Ask a question:")

                if user_input:
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    with st.spinner("Processing..."):
                        response, source = process_query(user_input)
                        st.session_state.messages.append({"role": "assistant", "content": response, "source": source})
                    save_chat_history(st.session_state.username, st.session_state.messages)
                    st.rerun()
        
        with tab2:
            with st.expander("📚 Manage Documents", expanded=True):
                st.subheader("➕ Add a New Document")
                uploaded_file = st.file_uploader("📚 Choose a PDF or TXT file", type=["pdf", "txt"])
                new_doc = st.text_area("Alternatively, enter document text manually:")

                if uploaded_file:
                    document_text = extract_text_from_file(uploaded_file)
                    if document_text:
                        wrapped_text = [wrap_text(content) for content in document_text]
                        for i, content in enumerate(wrapped_text):
                            new_id = f"doc_{collection.count() + 1}_{i}"
                            save_embeddings([content], [new_id])
                        st.success("Document added successfully!")
                        
                        if wordcloud_available:
                            with st.expander("📊 Document Visualization"):
                                combined_text = " ".join(wrapped_text)
                                wordcloud_fig = create_wordcloud(combined_text)
                                if wordcloud_fig:
                                    st.pyplot(wordcloud_fig)
                        else:
                            st.info("Word cloud visualization is not available. Install the 'wordcloud' package to enable this feature.")
                
                elif new_doc.strip():
                    new_id = f"doc_{collection.count() + 1}"
                    save_embeddings([new_doc], [new_id])
                    st.success("Document added successfully!")
        
        with tab3:
            st.subheader("📊 Analytics")
            if st.session_state.messages:
                df = pd.DataFrame([
                    {
                        'timestamp': msg.get('timestamp', ''),
                        'role': msg['role'],
                        'source': msg.get('source', ''),
                        'content_length': len(msg['content'])
                    }
                    for msg in st.session_state.messages
                ])
                
                if not df.empty:
                    st.write("Chat Statistics:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Messages", len(df))
                        st.metric("User Messages", len(df[df['role'] == 'user']))
                    with col2:
                        st.metric("AI Responses", len(df[df['role'] == 'assistant']))
                        sources = df[df['source'].notna()]['source'].value_counts()
                        st.write("Information Sources:", sources.to_dict())

async def run_telegram_bot():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_click))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    await application.initialize()
    
    try:
        logger.info("Starting bot...")
        await application.start()
        await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"Error in Telegram bot: {e}")
        
    finally:
        logger.info("Stopping bot...")
        await application.stop()
        await application.shutdown()

# --- Streamlit App Runner ---
def run_streamlit():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    os.system(f"streamlit run {__file__} --server.port=8501")

# --- Main Function ---
async def main():
    init_db()
    
    streamlit_process = threading.Thread(target=run_streamlit)
    streamlit_process.daemon = True 
    streamlit_process.start()
    
    try:
        await run_telegram_bot()
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        streamlit_app()
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
        except Exception as e:
            logger.error(f"Application error: {e}")