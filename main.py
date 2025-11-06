# ==============================================================================
# 1. IMPORTS
# ==============================================================================
from fastapi import FastAPI, HTTPException
import pandas as pd
from thefuzz import process, fuzz
from pandas import isna
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os # Added to make file paths more robust


# 2. CREATE FASTAPI APP

app = FastAPI(
    title="Deel Transaction Matching API",
    description="An API to match transactions to users using fuzzy logic and semantic search."
)


# 3. LOAD DATA & MODELS (GLOBAL)


# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSACTIONS_PATH = os.path.join(BASE_DIR, "transactions.csv")
USERS_PATH = os.path.join(BASE_DIR, "users.csv")

print("Loading data files and AI models...")
try:
    # --- Part 1: Load CSV Data ---
    transactions_df = pd.read_csv(TRANSACTIONS_PATH)
    users_df = pd.read_csv(USERS_PATH)
    
    users_df = users_df.dropna(subset=['name'])
    ALL_USER_NAMES = users_df['name'].tolist()
    
    # Set index for users_df for fast lookup
    users_df = users_df.set_index('id')
    
    # Set index for transactions_df for fast .loc[] lookup
    transactions_df = transactions_df.set_index('id')
    
    # --- Part 2: Prepare Data for Embedding ---
    # Fill 'NaN' (missing) descriptions with an empty string
    transactions_df['description'] = transactions_df['description'].fillna('')
    all_descriptions = transactions_df['description'].tolist()

    # --- Part 3: Load AI Model & Tokenizer ---
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    # This downloads and caches the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = model.tokenizer
    print("Model loaded.")

    # --- Part 4: Pre-calculate All Embeddings ---
    print(f"Calculating embeddings for {len(all_descriptions)} descriptions...")
    # This turns every description into a 384-dimension vector (number list)
    ALL_EMBEDDINGS = model.encode(all_descriptions, show_progress_bar=True)
    print("Embeddings calculated successfully.")

    # --- Part 5: Pre-calculate Token Counts ---
   
    token_counts = [len(tokenizer.encode(d)) for d in all_descriptions]
    transactions_df['token_count'] = token_counts
    
    print("Data and models loaded successfully.")

except FileNotFoundError as e:
    print(f"ERROR: Could not find file. Make sure 'transactions.csv' and 'users.csv' are in the same folder as main.py.")
    print(f"File not found error details: {e}") 
    transactions_df = None
    users_df = None
    ALL_USER_NAMES = []
    ALL_EMBEDDINGS = None



# 4. API ENDPOINT (TASK 1 - FUZZY MATCH)

@app.get("/match_users/{transaction_id}")
async def match_user_by_transaction_id(transaction_id: str):
    """
    Finds the best-matching user from 'users.csv'
    for a given transaction ID using fuzzy name matching.
    """
    
    # --- Step 1: Find the Transaction ---
    try:
        transaction = transactions_df.loc[transaction_id]
        description = transaction['description']
    except KeyError:
        # This ID wasn't in our index
        raise HTTPException(status_code=404, detail="Transaction ID not found")
    
    # --- Step 2: Handle Missing Data ---
    if isna(description) or description == '':
        raise HTTPException(
            status_code=400,
            detail="Transaction has no description to match against."
        )


    best_match = process.extractOne(
        description,
        ALL_USER_NAMES,
        scorer=fuzz.token_sort_ratio
    )
    
    matched_name, match_score = best_match[0], best_match[1]
    
    # --- Step 4: Get Full User Details ---
    # Find the user's row by their name to get their ID
    matched_user = users_df[users_df['name'] == matched_name].iloc[0]

    # --- Step 5: Return the Final JSON Response ---
    return {
        "task": "Task 1 - Fuzzy Matching",
        "transaction_id": transaction_id,
        "transaction_description": description,
        "match_details": {
            "matched_user_id": matched_user.name, # .name is the index (the ID)
            "matched_user_name": matched_name,
            "match_score": match_score
        }
    }


# 5. API ENDPOINT (TASK 2 - SEMANTIC SEARCH)


class SimilarityRequest(BaseModel):
    transaction_id: str
    k: int = 5 # 'k' is the number of results, default to 5

# '@app.post' creates a POST endpoint
@app.post("/find_similar_transactions")
async def find_similar_transactions(request: SimilarityRequest):

    
    # --- Step 1: Get the Transaction & its Pre-calculated Embedding ---
    try:
        # Find the integer row number (iloc) for the transaction ID
        transaction_index = transactions_df.index.get_loc(request.transaction_id)
        
        # Get the original transaction details
        transaction = transactions_df.iloc[transaction_index]
        
        # Get the pre-calculated embedding using that row number
        embedding_to_match = ALL_EMBEDDINGS[transaction_index].reshape(1, -1)
        
    except KeyError:
        raise HTTPException(status_code=404, detail="Transaction ID not found")
        

    scores = cosine_similarity(embedding_to_match, ALL_EMBEDDINGS)
    
    # The result is 2D, so we flatten it to a 1D list
    scores = scores.flatten()

    
    sorted_indices = np.argsort(scores)
    

    top_k_indices = sorted_indices[-request.k-1:-1][::-1]

    # --- Step 4: Format the Results ---
    results = []
    for i in top_k_indices:
        match = transactions_df.iloc[i]
        results.append({
            "transaction_id": match.name, # .name is the index (the ID)
            "description": match['description'],
            "similarity_score": float(scores[i]), # Get the score
            "token_count": int(match['token_count']) # Get the token count
        })
        
    # --- Step 5: Return the Final JSON Response ---
    return {
        "task": "Task 2 - Semantic Similarity",
        "query_transaction": {
            "transaction_id": request.transaction_id,
            "description": transaction['description'],
            "token_count": int(transaction['token_count'])
        },
        "total_tokens_used": int(transaction['token_count']),
        "similar_matches": results
    }

