input_path="../../datasets/chunked_documents/esa_documents/chunking_cks_1024_ovl_8.json"
output_path="../../datasets/example_questions/esa_queries_llama3_70b_cks_512.json"
model_name="meta-llama/Meta-Llama-3-70B-Instruct"
chunk_size=512
python ragas_generation_esa.py --input_path $input_path \
                      --output_path $output_path \
                      --model_name $model_name \
                      --chunk_size $chunk_size