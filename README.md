# Spend Insights Assistant

An AI-powered chatbot that answers natural-language questions about credit card transaction data, using a RAG (Retrieval-Augmented Generation) pipeline with a local LLM via Ollama. Answers are grounded in real data, with charts rendered automatically alongside relevant answers.

## Tech stack
Python, Pandas, ChromaDB (vector store), Ollama (local LLM), Streamlit (UI), Plotly (charts)

## Features
- Automatic chart generation based on question type (trend, category, segmentation)
- Light/dark theme toggle
- Runs entirely locally, no API key or cost

## Setup
1. Install Ollama: https://ollama.com/download
2. ollama pull llama3.2:1b
3. pip install -r requirements.txt
4. streamlit run app.py
