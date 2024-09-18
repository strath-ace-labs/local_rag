import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os

def calculate_similarities(result_directory, ground_truth_file, model_name):
    # Load ground truth data
    df_gt = pd.read_json(ground_truth_file)
    
    # Initialize the sentence transformer model
    embeddings = SentenceTransformer(model_name, trust_remote_code=True)
    
    # Encode ground truth outputs
    gt_embedding = embeddings.encode(df_gt.output)
    
    # Get all .json files from the directory
    result_files = [f for f in os.listdir(result_directory) if f.endswith('.json')]
    
    for result_file in result_files:
        file_path = os.path.join(result_directory, result_file)
        df_test = pd.read_json(file_path)
        
        # Encode test outputs
        if result_file == "results_phi_cot_true_threeshot_long.json":
            test_embedding = embeddings.encode(df_test.output[:10])
            gt_embedding_slice = gt_embedding[:10]
        else:
            test_embedding = embeddings.encode(df_test.output)
            gt_embedding_slice = gt_embedding
        
        # Calculate cosine similarities
        sims = util.cos_sim(test_embedding, gt_embedding_slice)
        scores = np.diagonal(sims)
        
        # Print results
        print(f"File: {result_file}")
        print(f"Mean similarity score: {scores.mean():.4f}")
        print(f"Max similarity score: {scores.max():.4f}")
        print(f"Index of max similarity: {scores.argmax()}")
        print("--------------------")

# Define input directory and model
result_directory = "./results"  # Replace with the actual path
ground_truth_file = "./results/results_llama3_70b_zeroshot.json"
model_name = "BAAI/bge-large-en-v1.5"

# Run the similarity calculation
calculate_similarities(result_directory, ground_truth_file, model_name)