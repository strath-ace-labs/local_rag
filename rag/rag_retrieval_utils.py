import torch 
from sentence_transformers import util
import json 
import os 
import numpy as np

def semantic_search(query, docs, model_emb, top_k=10, verbose = False):

    query_embedding = model_emb.encode(query, convert_to_tensor=True)
    embeds = [doc['embedding'] for doc in docs]

    ## transfer list of list to pytorch tensor
    embeds = torch.tensor(embeds).to(model_emb.device)    
    cos_scores = util.cos_sim(query_embedding, embeds)[0]
    
    ## torch order cos_scores
    #top_k = 10
    #top_results = torch.topk(cos_scores, k=top_k)
    top_results = torch.topk(cos_scores, k=cos_scores.shape[0])
    if verbose: 
       
        print("======================")
        print("Input query:", query)
        print(f"\nTop most similar entries in Knowledge Graph:")

        # print(names)

        for score, idx in zip(top_results[0], top_results[1]):
        #for score, name in zip(top_results[0], names):
            # print(iids[idx], "(Score: {:.4f})".format(score))
            print(docs[idx]['text'], "(Score: {:.4f})".format(score))

    extracts = [docs[idx] for idx in top_results.indices.tolist()]

    return top_results, extracts

import os 

def load_data(input_path):

    with open(input_path, 'r') as json_file:
        data = json.load(json_file)
    # Split the file path into the root and extension
    file_root, file_ext = os.path.splitext(input_path)
    print(file_root)
    embeddings = np.load(f'{file_root}_embeddings.npy')
    for counter,embed in enumerate(embeddings): 
        data[counter]['embedding'] = embed.tolist()
    
    return data