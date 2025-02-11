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
git clone https://github.com/your-repo/knowledge-navigator.git
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

---

## **Screenshots**
![image](https://github.com/user-attachments/assets/b93bee2e-a41c-4fad-88af-a5567a27825d)
![image](https://github.com/user-attachments/assets/27e00bd9-7ea9-4597-a243-5896ebb51fc0)
![image](https://github.com/user-attachments/assets/4365d81e-39da-4e1a-992d-0044155c262a)
![image](https://github.com/user-attachments/assets/cdd552f5-a577-42df-8a00-a0fb703262a0)
![image_2025-02-11_16-22-08](https://github.com/user-attachments/assets/e7fdead5-ca60-40e5-8fd8-bce513e9249a)
![image_2025-02-11_16-22-28](https://github.com/user-attachments/assets/7c85f82a-87e0-440b-ba4d-891247a64507)



---

