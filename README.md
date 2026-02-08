# 🤖 Banking & Legal RAG Chatbot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com)
[![Deployed on Render](https://img.shields.io/badge/Deployed-Render-purple.svg)](https://render.com)

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about Kazakhstan banking law and IFRS regulations. Built with LangChain, FAISS vector store, and deployed on Render with both web and Telegram interfaces.

## 🎯 Features

- **RAG Architecture**: Combines document retrieval with LLM generation for accurate, source-grounded answers
- **Multi-Model Support**: Works with both OpenAI GPT and Anthropic Claude APIs
- **Vector Search**: FAISS-powered semantic search across legal documents
- **Dual Interface**: Web API (FastAPI) + Telegram bot integration
- **Production Deployed**: Containerized with Docker, hosted on Render

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User Query     │────▶│  FastAPI Server  │────▶│  LangChain      │
│  (Web/Telegram) │     │                  │     │  RAG Pipeline   │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐     ┌────────▼────────┐
                        │  LLM Response    │◀────│  FAISS Vector   │
                        │  (GPT/Claude)    │     │  Store + Docs   │
                        └──────────────────┘     └─────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM Framework** | LangChain |
| **Vector Store** | FAISS |
| **Embeddings** | OpenAI Ada / HuggingFace |
| **API Server** | FastAPI |
| **Deployment** | Docker + Render |
| **Bot Interface** | python-telegram-bot |

## 📁 Project Structure

```
├── main.py                 # FastAPI application & RAG logic
├── build_index_local.py    # Script to build FAISS index from documents
├── telegram_bot.py         # Telegram bot integration
├── docs/                   # Source documents (banking laws, IFRS)
├── index/                  # Pre-built FAISS vector index
├── static/                 # Web assets
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
└── render.yaml             # Render deployment config
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key or Anthropic API key

### Installation

```bash
# Clone the repository
git clone https://github.com/daureny/Rag_gpt_bot_1.git
cd Rag_gpt_bot_1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"
# or
export ANTHROPIC_API_KEY="your-api-key"
```

### Running Locally

```bash
# Start the FastAPI server
python main.py

# Server runs at http://localhost:8000
```

### Building the Index (Optional)

If you want to add new documents:

```bash
# Place PDF/TXT documents in /docs folder
python build_index_local.py
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/ask` | POST | Submit a question |
| `/chat` | POST | Chat with context |

### Example Request

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the capital requirements for banks in Kazakhstan?"}'
```

## 🌐 Live Demo

The bot is deployed and accessible at:
- **Web**: [Render deployment URL]
- **Telegram**: [@YourBotName]

## 🔧 Configuration

Environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes* |
| `ANTHROPIC_API_KEY` | Claude API key | Yes* |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | For Telegram |

*One of OpenAI or Anthropic key required

## 📊 Use Cases

- **Legal Research**: Quick answers about Kazakhstan banking regulations
- **Compliance Queries**: IFRS 9 implementation questions
- **Document Q&A**: Query across multiple regulatory documents

## 🧠 How RAG Works

1. **Document Ingestion**: PDF/text documents are chunked and embedded
2. **Vector Storage**: Embeddings stored in FAISS for fast retrieval
3. **Query Processing**: User question is embedded and matched against documents
4. **Context Injection**: Relevant chunks are passed to LLM with the question
5. **Response Generation**: LLM generates answer grounded in retrieved context

## 📝 License

MIT License - feel free to use for your own projects.

## 👤 Author

**Dauren Yeleukenov**
- Finance & Risk Management Professional
- Python Developer specializing in FinTech & AI
- [LinkedIn](https://linkedin.com/in/yourprofile)

---

*Built with ❤️ using LangChain and FastAPI*
