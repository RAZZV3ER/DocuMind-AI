# DocuMind AI 🧠📄

DocuMind AI is an intelligent document analysis and RAG (Retrieval-Augmented Generation) agent application built with Flask, Groq LLM, Supabase, and hybrid semantic retrieval.

---

## 🚀 Features

- **Multi-document Analysis**: Upload PDFs and text documents.
- **Hybrid Retrieval Engine**: Combines BM25 keyword search with dense semantic embeddings (`sentence-transformers`) and Reciprocal Rank Fusion (RRF).
- **Agentic Workflow**: Dynamic tool use, smart query expansion, document summarization, and citation tracking powered by Groq.
- **User Authentication & Cloud Storage**: Secure session management and storage backed by Supabase.
- **Responsive Web Interface**: Sleek UI with auth, dashboard, document viewer, and conversational Q&A.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask, Flask-Session
- **LLM**: Groq API
- **Embeddings & Vector Search**: Sentence-Transformers, NumPy, Scikit-learn
- **Database & Storage**: Supabase
- **Frontend**: HTML5, CSS3, JavaScript (Jinja2 templates)

---

## 📦 Local Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/RAZZV3ER/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Update `.env` with your actual credentials:
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `SECRET_KEY`
- `GROQ_API_KEY`
- `GROQ_MODEL`

### 5. Run the application
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your web browser.

---

## 🌐 Deployment

This application includes a `Procfile` configured for deployment on platforms like [Render](https://render.com/) or [Railway](https://railway.app/):

```
web: gunicorn app:app
```

When deploying:
1. Connect your GitHub repository `RAZZV3ER/DocuMind-AI`.
2. Set build command to `pip install -r requirements.txt`.
3. Set start command to `gunicorn app:app`.
4. Add all environment variables from `.env.example` into your hosting provider's Environment Variables dashboard.
