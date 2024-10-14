import re 
import os 
from tqdm.notebook import tqdm
import argparse
from papermage.recipes import CoreRecipe
from transformers import AutoTokenizer
import pandas as pd 
import json 
import numpy as np

from pathlib import Path

from sentence_transformers import SentenceTransformer, util

def split_tokens_with_overlap(tokens, chunk_size, overlap):
    """
    function that splits a list of tokens into overlapping chunks of size chunk_size.
    LLMs have limited context length, so we need to split the input into chunks.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
        
    chunks = []
    start = 0
    end = chunk_size
    
    while start < len(tokens):
        chunk = tokens[start:end]
        chunks.append(chunk)
        start += (chunk_size)
        if start > len(tokens):
            break
        start -=  overlap
        end = start + chunk_size
    
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help='model name')
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument('--input_path', type=str, help='input path')
    parser.add_argument('--output_path', type=str, help='output path')
    parser.add_argument('--chunk_size', type=int, default=1024)
    parser.add_argument('--overlap', type=int, default=8)
    args = parser.parse_args()

    input_path = args.input_path
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    output_path = args.output_path
    # Ensure the output directory exists
    output_file_path = Path(output_path)
    if not output_file_path.parent.exists():
        try:
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Created directory {output_file_path.parent}")
        except Exception as e:
            print(f"Failed to create directory {output_file_path.parent}: {e}")
            exit(1)
    ## Loading papermage core recipe to extract text from pdf files 
    recipe = CoreRecipe()

    ## parameters for chunking
    chunk_size = args.chunk_size
    overlap = chunk_size // 8

    paragraphs = []
    print("Extracting text chunks from PDFs...")

    ## extract the text from the pdf files and split them into smaller chunks 
    for file in tqdm(os.listdir(input_path)):

        #try:
        if file.endswith(".pdf"):
            
            ## join input path with file name to get full path 
            file_path = os.path.join(input_path, file)

            ## run papermage recipe to get document object
            doc = recipe.run(file_path)
            ## Joins all pages into one text 
            full_text = "".join([par.text for par in doc.pages])                  
            tokenized_input = tokenizer(full_text, add_special_tokens=False)
            
            ## chunks the input into smaller chunks 
            chunks = split_tokens_with_overlap(tokenized_input['input_ids'], chunk_size, overlap)
            for chunk in chunks:
                paragraphs.append({"document": doc.titles[0].text, "text": tokenizer.decode(chunk)})

        elif file.endswith(".txt"):
            
            ## join input path with file name to get full path 
            file_path = os.path.join(input_path, file)

            with open(file_path, 'r') as f:
                full_text = f.read()
            
            tokenized_input = tokenizer(full_text, add_special_tokens=False)

            ## chunks the input into smaller chunks
            chunks = split_tokens_with_overlap(tokenized_input['input_ids'], chunk_size, overlap)
            
            document_title = str(file.split(".")[0])
            
            for chunk in chunks:
                paragraphs.append({"document": document_title, "text": f"Document: {document_title}\n"+tokenizer.decode(chunk)})

        else: 
            print(file)
                
        # except Exception as e:
        #     print("Exception Occured: ", file)
        #     pass
        
        ## save after every doc 
        df = pd.DataFrame(paragraphs)
        df.to_json(args.output_path, orient="records", indent= 4)

    ## add embeddings to the file 
    print("Embedding Chunks...")
    # Load from JSON file
    with open(args.output_path, 'r') as json_file:
        data = json.load(json_file)
        
    ## extracts saved chunk texts
    chunk_texts = []
    for i in range(len(data)): 
        chunk_texts.append(data[i]['text'])

    ## Embed the chunks
    model_emb = SentenceTransformer(args.embedding_model)
    embeddings = model_emb.encode(chunk_texts)

    # Split the file path into the root and extension
    file_root, file_ext = os.path.splitext(args.output_path)
    print(file_root)

    # Save embeddings as a binary npy file for fast loading later 
    np.save(f'{file_root}_embeddings.npy', embeddings)

    print("DONE!")

if __name__ == "__main__":
    main()
    
