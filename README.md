# DocuChat

DocuChat is a grounded, retrieval-augmented generation (RAG) document assistant that lets users upload PDFs and chat with them using FastAPI, Streamlit, LangChain, Chroma, and Google Gemini.

> **Note**: This project is currently under construction (Phase 1: Foundation & Settings).

## Prerequisites
- Python 3.10 to 3.14 (64-bit)
- Git
- Google AI Studio API key (obtain from [Google AI Studio](https://aistudio.google.com/))

## Setup Instructions

1. **Clone the repository and navigate to the project root**:
   ```bash
   cd "DocuChat Project"
   ```

2. **Create a virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   Open `.env` in your text editor and add your `GOOGLE_API_KEY`.
