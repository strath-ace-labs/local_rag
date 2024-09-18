input_path="../../datasets/chunked_documents/esa_documents/chunking_cks_1024_ovl_8.json"
output_path="../../datasets/example_questions/esa_queries_llama3_8b_cks_1024.json"
python ragas_generation_esa.py --input_path $input_path \
                      --output_path $output_path \