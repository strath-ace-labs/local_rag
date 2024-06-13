

model_name="meta-llama/Meta-Llama-3-8B-Instruct"
input_path="./datasets/esa_documents"
output_path="./datasets/chunked_documents/esa_documents/chunking_cks_1024_ovl_8.json"
embeddings_model="BAAI/bge-large-zh-v1.5"
chunk_size=1024
python preprocessing/extract_text_and_chunk_pdfs.py --model_name $model_name \
                                      --input_path $input_path \
                                      --output_path $output_path \
                                      --embedding_model $embeddings_model \
                                      --chunk_size $chunk_size
                                      