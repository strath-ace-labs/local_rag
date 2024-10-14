
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
input_path="./datasets/nasa_teaching_transcripts/transcript"
output_path="./datasets/chunked_documents/nasa_teaching_spacesuit/chunking_cks_1536_ovl_8.json"
embeddings_model="BAAI/bge-large-en-v1.5"
chunk_size=1536

python preprocessing/extract_text_and_chunk_pdfs.py --model_name $model_name \
                                      --input_path $input_path \
                                      --output_path $output_path \
                                      --embedding_model $embeddings_model \
                                      --chunk_size $chunk_size

