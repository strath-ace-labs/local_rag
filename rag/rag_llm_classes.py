from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import GenerationConfig, AutoModelForCausalLM
import torch 
import json
from transformers import BitsAndBytesConfig
## load tokenizer and model via transformers 
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import GenerationConfig, AutoModelForCausalLM

from llama_cpp import Llama
import llama_cpp.llama_tokenizer

from huggingface_hub import hf_hub_download

from datetime import datetime
import os
from pathlib import Path

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [],tokenizer= None, encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if self.tokenizer.decode(stop) == self.tokenizer.decode(last_token):
                return True
        return False

## Class to run inference with Huggingface models on GPU
class HuggingfaceLLM():
    def __init__(self, model, tokenizer, device, generate_config_dict=None, stopping_criteria=None):
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generate_config_dict = generate_config_dict

        if generate_config_dict is None:
            generate_config_dict = {
                "max_new_tokens": 32,
                "do_sample": True,
                "top_p": 0.6,
                "top_k": 0,
                "temperature": 0.9,
            }

        self.stopping_criteria = stopping_criteria

        self.json_log_file = "../datasets/logging/log_llm_calls.json"
        self.json_log_file = Path(self.json_log_file)
        self.current_episode_messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            }
        ]
    def get_system_prompt(self) -> str:
        # Starting system prompt or any initialization text if needed
        return "You are an helpful engineering expert. Try to answer the question with the provided context as best as you can."

    def create_prompt(self, question, contexts):
        
        message = f"""CONTEXT: """
        
        for context in contexts: 
            message += f"""{context}\n___________________\n"""
        
        message += f"""QUESTION: {question}"""

        self.current_episode_messages = [{"role": "user", "content": message}]

        start_assistant_message= " ANSWER:"
        
        self.current_episode_messages.append({"role": "assistant", "content": start_assistant_message})

        prompt = self.tokenizer.apply_chat_template(
                    self.current_episode_messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False)

        #print(prompt)
        prompt = prompt.strip()
        prompt = prompt[:-len(self.tokenizer.eos_token)]
        #print(prompt)
        prompt = prompt[len(self.tokenizer.bos_token):]
        #print(prompt)
        return prompt
    
    def parse_output(self, prompt, output):

        output =output[len(self.tokenizer.decode(self.tokenizer.encode(prompt))):]
        return output
    
    def llm(self,
        prompt,
        ):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
             **self.generate_config_dict,
         )    
        with torch.no_grad():
            
            generation_output = self.model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                generation_config=generation_config,
                stopping_criteria=self.stopping_criteria,
                #output_hidden_states= True,
                #output_scores=True,
                #output_attentions=True,
            )
            s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)

        return output

    def log_to_json(self, data):
        
        if not self.json_log_file.exists():
            try:
                self.json_log_file.parent.mkdir(parents=True, exist_ok=True)
                with self.json_log_file.open('w') as f:
                    json.dump([], f)
            except Exception as e:
                print(f"Error creating log file: {e}")

        try:
            with self.json_log_file.open('r+') as f:
                log_entries = json.load(f)
                log_entries.append(data)
                f.seek(0) # moves the file pointer to the beginning of the file
                json.dump(log_entries, f, indent=4)
        
        except Exception as e:
            print(f"Error writing to log file: {e}")

    def act(self, question, context):
        
        prompt = self.create_prompt(question, context)
        #print(prompt)
        response = self.llm(prompt)
        response = self.parse_output(prompt, response)

        # Collect metadata
        log_entry = {
            "question": question,
            "prompt": prompt,
            "context": context,
            "response": response,
            "date_created": datetime.now().isoformat(),
            "model": self.model.config._name_or_path,
            "generate_config_dict": self.generate_config_dict,
            "device": self.device
        }

        #print(log_entry)
        # Log to JSON file
        self.log_to_json(log_entry)

        self.current_episode_messages.append({"role": "assistant", "content": response})

        return response
    
## Class to run inference with Llamacpp models on CPU
class LlamacppLLM(HuggingfaceLLM):

    def __init__(self, model, tokenizer, generate_config_dict=None, stopping_criteria=None):
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = 'cpu'
        self.generate_config_dict = generate_config_dict
        
        self.stopping_criteria = [self.tokenizer.eos_token]

        self.json_log_file = "../datasets/logging/log_llm_calls.json"
        self.json_log_file = Path(self.json_log_file)

        self.current_episode_messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(),
            }
        ]

    def llm(self,
        prompt,
        ):

        output = self.model(
                prompt, # Prompt
                max_tokens=600, # Generate up to 32 tokens, set to None to generate up to the end of the context window
                stop=self.stopping_criteria, # Stop generating just before the model would generate a new question
                echo=True # Echo the prompt back in the output
            ) # Generate a completion, can also call create_completion

        output = output['choices'][0]['text']   

        return output
    
    def parse_output(self, prompt, output):

        output =output[len(self.tokenizer.decode(self.tokenizer.encode(prompt, add_special_tokens = False))):]
        return output
    
    def act(self, question, context):
        
        prompt = self.create_prompt(question, context)
        response = self.llm(prompt)

        #print(response)
        response = self.parse_output(prompt, response)

        self.current_episode_messages.append({"role": "assistant", "content": response})

        # Collect metadata
        log_entry = {
            "question": question,
            "prompt": prompt,
            "context": context,
            "response": response,
            "date_created": datetime.now().isoformat(),
            "model": self.tokenizer.name_or_path,
            "generate_config_dict": self.generate_config_dict,
            "device": self.device
        }

        # Log to JSON file
        self.log_to_json(log_entry)

        return response
    
def load_cpu_model(model_name="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF", filename="Meta-Llama-3-8B-Instruct.Q5_K_M.gguf"):
    script_dir = "../"
    output_folder = f"{script_dir}cpu_models/"
    if not os.path.exists(f"{output_folder}{filename}"):
        print("Downloading model...")
        output_file = output_folder
        url =hf_hub_download(model_name, filename= filename, local_dir = output_file)

    tokenizer_dict = {"Meta-Llama-3-8B-Instruct.Q5_K_M.gguf" : "meta-llama/Meta-Llama-3-8B-Instruct"}

    model = Llama(
       model_path=f"{script_dir}cpu_models/{filename}",
        tokenizer= llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(tokenizer_dict[filename]),
        # n_gpu_layers=-1, # Uncomment to use GPU acceleration
        # seed=1337, # Uncomment to set a specific seed
        n_ctx=4096, # Uncomment to increase the context window
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dict.get(filename, "meta-llama/Meta-Llama-3-8B-Instruct"))

    llm = LlamacppLLM(model, tokenizer)

    return llm 

def load_gpu_model(model_name = "meta-llama/Meta-Llama-3-8B-Instruct", device='cuda', quantization_config = None):

    MAX_NEW_TOKENS=600

    generation_args = {"max_new_tokens":MAX_NEW_TOKENS,
                "do_sample": True, 
                "num_beams" : 1, 
                "num_return_sequences" : 1, 
                "temperature": 0.2,# 0.8, 
                "top_p": 0.95,
                "min_new_tokens": 256, 
                "begin_suppress_tokens": [128009], 
                }

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code = True,
                    quantization_config= quantization_config
                #device_map="auto",
                )   
    #print(model.device) 
    if model.device.type == 'cpu':
        
        model.to(device)
        
    stop_words = [[tokenizer.eos_token]]#, "yes", "no"] #"\nYes"
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, tokenizer=tokenizer)])
    llm = HuggingfaceLLM(model, tokenizer, device, generate_config_dict=generation_args, stopping_criteria=stopping_criteria)
    
    return llm
    
def load_inference_model(**keyargs):
    """
    Depending on availability of system resources loads either a GPU or CPU model
    
    """
    device = keyargs.get('device', None)
    
    if device == None:

        device = ("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda":
        
        print("Loading GPU model")
        gpu_model = keyargs.get('model_name', "meta-llama/Meta-Llama-3-8B-Instruct")

        ## Calculates available GPU VRAM and loads a GPU model if there is enough VRAM available.
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert bytes to GB
        reserved_vram = torch.cuda.memory_reserved(0) / (1024 ** 3)  # Convert bytes to GB
        allocated_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)  # Convert bytes to GB
        free_vram = total_vram - reserved_vram
        
        if free_vram > 16: 
            print("Loading GPU model")
            llm = load_gpu_model(model_name=gpu_model)
        elif free_vram > 12: 
            print("Loading 8-bit quantized model")
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            llm = load_gpu_model(model_name=gpu_model, quantization_config = quant_config)
        
        elif free_vram > 8: 
            print("Loading 4-bit quantized model")
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            llm = load_gpu_model(model_name=gpu_model, quantization_config = quant_config)
        
        else:
            print("Fallback to CPU model, not enough VRAM available")
            
            llm = load_cpu_model()

    else:

        cpu_model = keyargs.get('model_name', "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF")
        file_name = keyargs.get('file_name', "Meta-Llama-3-8B-Instruct.Q5_K_M.gguf")
        llm = load_cpu_model(model_name=cpu_model, filename=file_name)
        
    return llm