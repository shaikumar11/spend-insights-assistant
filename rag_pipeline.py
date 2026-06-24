"""
AI-Powered Customer Spend Insights Assistant (FREE / LOCAL VERSION)
---------------------------------------------------------------------
Same RAG (Retrieval-Augmented Generation) pipeline as the OpenAI version,
but uses Ollama to run a free, open-source LLM locally on your own machine.
No API key, no billing, no internet required after the one-time model download.

Setup (do this BEFORE running this script):
    1. Install Ollama from https://ollama.com/download
    2. Open a terminal and run: ollama pull llama3.2:1b
       (downloads a ~1.3GB model — smaller and faster than the full llama3.2,
       a good fit for this kind of structured lookup/summary task)
    3. Leave Ollama running in the background (it starts automatically on
       most installs, or run: ollama serve)

Then:
    pip install -r requirements.txt
    python rag_pipeline.py
"""

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import requests

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
DATA_PATH = "transactions.csv"
COLLECTION_NAME = "spend_summaries"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"   # smaller/faster than llama3.2 (3b). Swap to "phi3:mini" or
                                # back to "llama3.2" for more accuracy at the cost of speed.


def call_ollama(prompt: str) -> str:
    """Send a prompt to the local Ollama server and return the generated text
    (blocking — waits for the full response). Use call_ollama_stream() instead
    if you want to show text as it's generated."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to Ollama at http://localhost:11434. "
            "Make sure Ollama is installed and running (try opening a new "
            "terminal and running: ollama serve)."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Ollama returned an error: {e}. "
            f"Make sure you've pulled the model first: ollama pull {OLLAMA_MODEL}"
        )


def call_ollama_stream(prompt: str):
    """Generator version: yields text chunks as Ollama generates them, so the
    UI can show a typing effect instead of one long silent wait. Use this
    inside st.write_stream() in the Streamlit app."""
    try:
        with requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
            timeout=120,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = __import__("json").loads(line)
                if "response" in chunk:
                    yield chunk["response"]
                if chunk.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to Ollama at http://localhost:11434. "
            "Make sure Ollama is installed and running (try opening a new "
            "terminal and running: ollama serve)."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Ollama returned an error: {e}. "
            f"Make sure you've pulled the model first: ollama pull {OLLAMA_MODEL}"
        )


# ---------------------------------------------------------------------------
# 2. BUILD SUMMARY DOCUMENTS FROM RAW TRANSACTIONS
# ---------------------------------------------------------------------------
def build_summary_documents(df: pd.DataFrame) -> list:
    """
    Aggregate raw transaction rows into short natural-language summaries.
    Each summary becomes one retrievable 'document' for the RAG store.

    Always builds category/month summaries. If the richer dashboard columns
    are present (customer_segment, merchant_name, city), also builds summaries
    for those — so the assistant can correctly answer segment/merchant/city
    questions instead of saying the data isn't available when it actually is.

    Expected columns in transactions.csv:
        transaction_date, customer_id, merchant_category, amount
    Optional richer columns (transactions_dashboard.csv):
        customer_segment, merchant_name, city, card_type, channel
    """
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["month"] = df["transaction_date"].dt.to_period("M").astype(str)

    documents = []
    doc_id = 0

    # --- Month TOTAL summaries (across all categories) — added so questions
    # like "which month had the highest spending" are a direct lookup rather
    # than something the model has to add up itself across category snippets ---
    month_totals = (
        df.groupby("month")
        .agg(total_spend=("amount", "sum"), transaction_count=("amount", "count"))
        .reset_index()
        .sort_values("month")
    )
    overall_total = month_totals["total_spend"].sum()
    highest_month_row = month_totals.loc[month_totals["total_spend"].idxmax()]
    lowest_month_row = month_totals.loc[month_totals["total_spend"].idxmin()]

    for _, row in month_totals.iterrows():
        text = (
            f"In {row['month']} (overall, across all categories combined), total spend was "
            f"${row['total_spend']:.2f} across {int(row['transaction_count'])} transactions."
        )
        documents.append({"id": f"doc_{doc_id}", "text": text})
        doc_id += 1

    # One explicit comparison document — directly answers "which month was highest/lowest"
    comparison_text = (
        f"Across the full period, total spend was ${overall_total:.2f}. "
        f"The highest-spending month overall was {highest_month_row['month']} "
        f"with ${highest_month_row['total_spend']:.2f} in total spend. "
        f"The lowest-spending month overall was {lowest_month_row['month']} "
        f"with ${lowest_month_row['total_spend']:.2f} in total spend."
    )
    documents.append({"id": f"doc_{doc_id}", "text": comparison_text})
    doc_id += 1

    # --- Category x month summaries (always available) ---
    grouped = (
        df.groupby(["month", "merchant_category"])
        .agg(total_spend=("amount", "sum"),
             transaction_count=("amount", "count"),
             avg_transaction=("amount", "mean"))
        .reset_index()
    )
    for _, row in grouped.iterrows():
        text = (
            f"In {row['month']}, the '{row['merchant_category']}' category had "
            f"{int(row['transaction_count'])} transactions totaling "
            f"${row['total_spend']:.2f}, with an average transaction value of "
            f"${row['avg_transaction']:.2f}."
        )
        documents.append({"id": f"doc_{doc_id}", "text": text})
        doc_id += 1

    # --- Overall category totals (across the whole period, not per-month) —
    # added so "which category had the highest spend overall" is also a
    # direct lookup, same reasoning as the month-total fix above ---
    category_totals = (
        df.groupby("merchant_category")
        .agg(total_spend=("amount", "sum"), transaction_count=("amount", "count"))
        .reset_index()
        .sort_values("total_spend", ascending=False)
    )
    top_category_row = category_totals.iloc[0]
    for _, row in category_totals.iterrows():
        text = (
            f"Overall (across the entire period), the '{row['merchant_category']}' category had "
            f"{int(row['transaction_count'])} transactions totaling ${row['total_spend']:.2f} in spend."
        )
        documents.append({"id": f"doc_{doc_id}", "text": text})
        doc_id += 1
    documents.append({
        "id": f"doc_{doc_id}",
        "text": (
            f"The highest-spending category overall, across the entire period and all months "
            f"combined, was '{top_category_row['merchant_category']}' with "
            f"${top_category_row['total_spend']:.2f} in total spend."
        ),
    })
    doc_id += 1

    # --- Customer segment summaries (only if this column exists) ---
    if "customer_segment" in df.columns:
        seg_grouped = (
            df.groupby("customer_segment")
            .agg(total_spend=("amount", "sum"),
                 transaction_count=("amount", "count"),
                 customer_count=("customer_id", "nunique"))
            .reset_index()
        )
        for _, row in seg_grouped.iterrows():
            text = (
                f"The '{row['customer_segment']}' customer segment includes "
                f"{int(row['customer_count'])} customers who made "
                f"{int(row['transaction_count'])} transactions totaling "
                f"${row['total_spend']:.2f} in spend."
            )
            documents.append({"id": f"doc_{doc_id}", "text": text})
            doc_id += 1

        top_segment_row = seg_grouped.loc[seg_grouped["total_spend"].idxmax()]
        documents.append({
            "id": f"doc_{doc_id}",
            "text": (
                f"Among the customer segments, the one with the highest total spend is "
                f"'{top_segment_row['customer_segment']}', with ${top_segment_row['total_spend']:.2f} "
                f"in total spend from {int(top_segment_row['customer_count'])} customers."
            ),
        })
        doc_id += 1

    # --- Top merchant summaries (only if this column exists) ---
    if "merchant_name" in df.columns:
        merch_grouped = (
            df.groupby("merchant_name")
            .agg(total_spend=("amount", "sum"), transaction_count=("amount", "count"))
            .reset_index()
            .sort_values("total_spend", ascending=False)
            .head(15)  # top merchants only, to keep the index focused
        )
        for _, row in merch_grouped.iterrows():
            text = (
                f"The merchant '{row['merchant_name']}' had "
                f"{int(row['transaction_count'])} transactions totaling "
                f"${row['total_spend']:.2f} in spend."
            )
            documents.append({"id": f"doc_{doc_id}", "text": text})
            doc_id += 1

        # Explicit top-merchant answer document. Semantic search over many
        # similar-sounding "merchant X had $Y" documents can surface a
        # mid-ranked one instead of the true highest (the question "highest
        # spend" is similar to ALL of them semantically) — so we add one
        # direct, unambiguous statement of the actual #1 merchant.
        top_merchant_row = merch_grouped.iloc[0]
        documents.append({
            "id": f"doc_{doc_id}",
            "text": (
                f"The merchant with the single highest total spend, out of all merchants "
                f"in the dataset, is '{top_merchant_row['merchant_name']}' with "
                f"${top_merchant_row['total_spend']:.2f} in total spend across "
                f"{int(top_merchant_row['transaction_count'])} transactions. "
                f"No other merchant has a higher total."
            ),
        })
        doc_id += 1

    # --- City summaries (only if this column exists) ---
    if "city" in df.columns:
        city_grouped = (
            df.groupby("city")
            .agg(total_spend=("amount", "sum"), transaction_count=("amount", "count"))
            .reset_index()
        )
        for _, row in city_grouped.iterrows():
            text = (
                f"Customers in {row['city']} made {int(row['transaction_count'])} "
                f"transactions totaling ${row['total_spend']:.2f}."
            )
            documents.append({"id": f"doc_{doc_id}", "text": text})
            doc_id += 1

        top_city_row = city_grouped.loc[city_grouped["total_spend"].idxmax()]
        documents.append({
            "id": f"doc_{doc_id}",
            "text": (
                f"The city with the highest total spend is {top_city_row['city']}, "
                f"with ${top_city_row['total_spend']:.2f} in total spend."
            ),
        })
        doc_id += 1

    return documents


# ---------------------------------------------------------------------------
# 3. EMBED + STORE IN CHROMA (still local, no API needed for this part)
# ---------------------------------------------------------------------------
def build_vector_store(documents: list):
    chroma_client = chromadb.Client()
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn
    )

    collection.add(
        ids=[d["id"] for d in documents],
        documents=[d["text"] for d in documents],
    )
    return collection


# ---------------------------------------------------------------------------
# 4. RETRIEVE + GENERATE GROUNDED ANSWER (using local Ollama model)
# ---------------------------------------------------------------------------
def answer_question(collection, question: str, top_k: int = 8) -> str:
    """Blocking version: returns the full answer as one string once generation
    is complete. Smaller top_k than earlier versions — the dedicated month/category
    'total' summary documents added upstream are concise and high-signal, so we
    don't need a large retrieval count to get an accurate answer, which also
    keeps the prompt (and therefore response time) smaller."""
    results = collection.query(query_texts=[question], n_results=top_k)
    retrieved_context = "\n".join(results["documents"][0])

    prompt = f"""You are a data analyst assistant. Answer the user's question
using ONLY the data summaries provided below. If the data does not contain
the answer, say so clearly instead of guessing.

DATA SUMMARIES:
{retrieved_context}

QUESTION: {question}

ANSWER:"""

    return call_ollama(prompt)


def answer_question_stream(collection, question: str, top_k: int = 8):
    """Streaming version: yields text chunks as they're generated, for use
    with st.write_stream() so the answer appears progressively instead of
    one long silent wait."""
    results = collection.query(query_texts=[question], n_results=top_k)
    retrieved_context = "\n".join(results["documents"][0])

    prompt = f"""You are a data analyst assistant. Answer the user's question
using ONLY the data summaries provided below. If the data does not contain
the answer, say so clearly instead of guessing.

DATA SUMMARIES:
{retrieved_context}

QUESTION: {question}

ANSWER:"""

    yield from call_ollama_stream(prompt)


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    docs = build_summary_documents(df)
    print(f"Built {len(docs)} summary documents from {len(df)} transactions.")

    collection = build_vector_store(docs)
    print("Vector store ready.\n")

    sample_questions = [
        "Which merchant category had the highest spend last month?",
        "Summarize spending trends over the last quarter.",
    ]
    for q in sample_questions:
        print(f"Q: {q}")
        print(f"A: {answer_question(collection, q)}\n")
