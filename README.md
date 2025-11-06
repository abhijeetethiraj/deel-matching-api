# Deel Transaction Matching API

This project is a FastAPI server designed to solve two key challenges with transaction data:

1.  **Fuzzy Name Matching:** Intelligently matches messy transaction descriptions to a clean user list.
2.  **Semantic Search:** Finds transactions that are _conceptually similar_ to a given transaction.

---

## 🛠️ Features

- **Endpoint 1: `GET /match_users/{transaction_id}`**

  - Uses "fuzzy logic" (`thefuzz` library) to find the best-matching user name from `users.csv` inside a transaction's description.
  - Ideal for linking a transaction (e.g., "Payment from Liam J. Johnson") to a user (e.g., "Liam Johnson").

- **Endpoint 2: `POST /find_similar_transactions`**
  - Uses a sentence-transformer AI model (`all-MiniLM-L6-v2`) to convert all transaction descriptions into vector "embeddings."
  - Accepts a `transaction_id` and `k` (number of results).
  - Returns the top `k` most _semantically similar_ transactions by comparing their embeddings using cosine similarity.
  - Also returns the `token_count` for the query.

---

## 🚀 How to Run

### 1. Prerequisites

- Python 3.8+
- `pip` (Python package installer)

### 2. Installation

1.  Clone or download this project folder.
2.  Open your terminal in the project folder.
3.  Install all required libraries from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Running the Server

1.  From your terminal, run the `uvicorn` server:
    ```bash
    uvicorn main:app --reload
    ```
2.  The server will start, load the CSV files, and pre-calculate all embeddings. This may take 10-20 seconds on first launch as it downloads the AI model.
3.  Wait for the message: `Data and models loaded successfully.`

### 4. Testing the API

Once the server is running, open your web browser and go to the built-in documentation page:

**[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

You can test both endpoints live from this page.
