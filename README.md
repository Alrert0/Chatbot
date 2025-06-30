# **Knowledge Navigator Bot**

## **Overview**
Knowledge Navigator is an AI-powered chatbot and document management system with an integrated knowledge base. It supports:
- **Conversational AI** powered by **Llama3**.
- **Retrieval-Augmented Generation (RAG)** using **ChromaDB**.
- **Google Search Integration** for web-based information retrieval.
- **PDF and TXT document ingestion** for knowledge storage.
- **Word cloud generation** for document visualization.
- **Chat history storage** with **SQLite** authentication.
- **Profanity filtering** for chat inputs and documents.
- **Telegram Bot** support for interactive messaging.

## **Features**
✅ **LLM-based chatbot**  
✅ **ChromaDB-powered knowledge retrieval**  
✅ **Google Search API integration**  
✅ **User authentication with SQLite**  
✅ **PDF & TXT document ingestion**  
✅ **Word cloud analytics for documents**  
✅ **Chat history storage & analytics**  
✅ **Telegram Bot support**  

---

## **Installation**
### **Prerequisites**
Ensure you have **Python 3.8+** installed.

### **Clone the Repository**
```bash
git clone https://github.com/Alrert0/Chatbot/tree/main
cd knowledge-navigator
```

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **Configuration**
### **1. Set up API Keys**
Update the following constants in the script:
- **SerpAPI Key**: Set your API key in `SERPAPI_KEY`
- **Telegram Bot Token**: Replace with your bot's token

### **2. Database Initialization**
Run the following command to create the user database:
```bash
python init_db.py
```

---

## **Usage**
### **Run the Web App**
```bash
streamlit run app.py
```
Access the chatbot via your browser.

### **Run the Telegram Bot**
```bash
python telegram_bot.py
```
Interact with the bot on Telegram.

---

## **How It Works**
### **1. Chatbot Mode**
- Users can ask questions.
- The bot searches **ChromaDB** and **Google Search**.
- The answer is generated using the **Llama3 model**.

### **2. Document Management**
- Upload **PDF or TXT files**.
- Extracts and embeds content in **ChromaDB**.
- Generates a **word cloud visualization**.

### **3. Telegram Bot**
- Users send queries via Telegram.
- The bot retrieves information and responds.

