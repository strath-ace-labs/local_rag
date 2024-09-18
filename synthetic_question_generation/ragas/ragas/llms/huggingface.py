from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig 
from transformers import StoppingCriteria, StoppingCriteriaList

import torch 

from ragas.llms.base import RagasLLM

from langchain.schema import LLMResult

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

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


class HuggingfaceLLM(RagasLLM):

    def __init__(self, llm: str = "", load_8bit: bool = True, ):#, user_prompt="", system_prompt=""):

        LOAD_8BIT = load_8bit
        tokenizer = AutoTokenizer.from_pretrained(llm)

        if llm == "meta-llama/Meta-Llama-3-70B-Instruct":
            
            # Define the max memory for each GPU
            max_memory = {
                0: "23GiB",  # GPU 0 
                1: "22GiB", # GPU 1 
                "cpu": "0GiB"
            }
            LAYER_LLAMA = [f"model.layers.{i}" for i in range(80)]

            # Define the custom device map based on your requirements
            custom_device_map = {}

            # self_attn.q_proj.weight: cuda:1
            # self_attn.k_proj.weight: cuda:1
            # self_attn.v_proj.weight: cuda:1
            # self_attn.o_proj.weight: cuda:1
            # mlp.gate_proj.weight: cuda:1
            # mlp.up_proj.weight: cuda:1
            # mlp.down_proj.weight: cuda:1
            # input_layernorm.weight: cuda:1
            # post_attention_layernorm.weight: cuda:1
            
            # Specify the layer mappings: Layers 0-44 to GPU 0, Layers 45 onwards to GPU 1
            for i in range(80):  # Assuming your model has 80 layers, adjust as needed
                if i < 42:
                    custom_device_map[f"model.layers.{i}"] = 0  # Assign to GPU 0
                else:
                    custom_device_map[f"model.layers.{i}"] = 1  # Assign to GPU 1

            # Include any other specific mappings if needed (e.g., for output layers)
            custom_device_map["model.embed_tokens.weight"]=0
            custom_device_map["model.norm.weight"] = 1       # Assign final normalization to GPU 1
            custom_device_map["lm_head"] = 1    # Assign output head to GPU 1
            
            self.huggingface_llm = AutoModelForCausalLM.from_pretrained(
                    llm,
                    #load_in_8bit=LOAD_8BIT,
                    load_in_4bit=LOAD_8BIT,
                    torch_dtype=torch.bfloat16,
                    device_map=custom_device_map, 
                    #device_map="auto",
                    trust_remote_code=True,
                    #max_memory=max_memory,  # Specifies the max memory per GPU
                )     
  
            for name, param in self.huggingface_llm.named_parameters():
                print(f"{name}: {param.device}")
        
        else: 

            self.huggingface_llm = AutoModelForCausalLM.from_pretrained(
                    llm,
                    #load_in_8bit=LOAD_8BIT,
                    #load_in_4bit=LOAD_8BIT,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    #max_memory=max_memory,  # Specifies the max memory per GPU
                )     

        self.tokenizer = AutoTokenizer.from_pretrained(llm)

        #tokenizer.eos_token_id = 128009
        if "Meta-Llama-3" in self.huggingface_llm.config._name_or_path:

            self.huggingface_llm.generation_config.pad_token_id = 128009
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.stopping_criteria = None
        
    def evaluate(self,
        prompt,
        stopping_criteria = None,
        device = 'cuda',
        **kwargs,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    
        #inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            **kwargs,
        )
        #print("Length input_prompt: ", len(input_ids[0]))
        with torch.no_grad():
            generation_output = self.huggingface_llm.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                stopping_criteria=stopping_criteria,
                output_hidden_states= True,
                output_scores=True,
            )
            s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)

        return output, generation_output


    @property
    def llm(self):
        return self.huggingface_llm
    
    def parse_output(self, output, prompt):

        #output = output[len(self.tokenizer.decode(self.tokenizer.encode(prompt))):]
        #return output
        encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
        output =output[len(self.tokenizer.decode(encoded_prompt, skip_special_tokens=False)):]
        return output
    
    #def create_prompt_string(self, prompt_reason="", suffix=""):
        #return f"{self.user_prompt}{prompt_reason}\n{self.system_prompt}{suffix}"
    
    def create_prompt_string(self, prompt, suffix=""):
        
        if suffix!= "": 
            chat_dict = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": suffix}
            ]
            prompt = self.tokenizer.apply_chat_template(chat_dict, tokenize=False, add_generation_prompt=False)
            prompt = prompt[:-len(self.tokenizer.eos_token)]
        
        else:    
            chat_dict = [
                {"role": "user", "content": prompt}
            ]
            prompt = self.tokenizer.apply_chat_template(chat_dict, tokenize=False, add_generation_prompt=True)
        
        return prompt
        
    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        temperature: float = 1e-8,
        callbacks: t.Optional[Callbacks] = None,
        max_new_tokens: int = 384,
        min_tokens: int = 20,
        repetition_penalty: float = 1.0,
        suffix: str = "",
        stop_words: t.Optional[list] = None,
        #generation_args: t.Optional[dict] = None,
    ) -> LLMResult:
        

        if stop_words != None: 

            stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
            self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, tokenizer=self.tokenizer)])

        generation_args = {"max_new_tokens":max_new_tokens,
                "do_sample": True, 
                "num_beams" : 1, 
                "num_return_sequences" : 1, 
                "temperature": 0.2,# 0.8, 
                "repetition_penalty": repetition_penalty,
                "top_p": 0.95}
        
        if len(prompts) > 1:
            raise NotImplementedError("Multiple prompts not yet supported")
        
        #print(prompts)
        output = ("",{"scores": torch.zeros(8196).long()})
        prompt = self.create_prompt_string(prompt=prompts[0].messages[0].content, suffix=suffix)
        #prompt = prompts[0].messages[0].content
        print("Prompt:")
        print(prompt)
        stop_condition = len(output[1]['scores']) >= generation_args['max_new_tokens']

        if max_new_tokens == 384: 
            while stop_condition:
                #
                
                output = self.evaluate(prompt, stopping_criteria = self.stopping_criteria, device = 'cuda', **generation_args) #model = self.llm, tokenizer = self.tokenizer,
                stop_condition = len(output[1]['scores']) >= generation_args['max_new_tokens'] or len(output[1]['scores']) < min_tokens
                #print("Too short: ", len(output[1]['scores']) < min_tokens)
                #print("Too long: ", len(output[1]['scores']) >= generation_args['max_new_tokens'])
                #print("Max_new_tokens: ", generation_args['max_new_tokens'])
                print("Re-generation condition: ", stop_condition)
                print("Length generated: " + str(len(output[1]['scores'])))
                generation_args['max_new_tokens'] = generation_args['max_new_tokens'] + 256
                if generation_args['max_new_tokens'] > 384: break  

        else: 
            output = self.evaluate(prompt, stopping_criteria = self.stopping_criteria, device = 'cuda', **generation_args) #model = self.llm, tokenizer = self.tokenizer,
            print("Length generated: " + str(len(output[1]['scores'])))
            #print(generation_args['max_new_tokens'])
        output = self.parse_output(output[0], prompt=prompt)
        output = suffix + output
        #print("Output: ")
        #print(output)
        substring_to_remove = self.tokenizer.eos_token #"<|end_of_turn|>"

        # Removing the substring
        output = output.replace(substring_to_remove, "")
        #print("______________")
        return LLMResult(generations= [[{'text':output}]])
    
    async def agenerate(
        self,
        prompt: ChatPromptTemplate,
        n: int = 1,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        pass
    #     temperature = 0.2 if n > 1 else 0
    #     if isBedrock(self.llm) and ("model_kwargs" in self.llm.__dict__):
    #         self.llm.model_kwargs = {"temperature": temperature}
    #     else:
    #         self.llm.temperature = temperature

    #     if self.llm_supports_completions(self.llm):
    #         self.langchain_llm = t.cast(
    #             MultipleCompletionSupportedLLM, self.langchain_llm
    #         )
    #         old_n = self.langchain_llm.n
    #         self.langchain_llm.n = n
    #         if isinstance(self.llm, BaseLLM):
    #             result = await self.llm.agenerate(
    #                 [prompt.format()], callbacks=callbacks
    #             )
    #         else:  # if BaseChatModel
    #             result = await self.llm.agenerate(
    #                 [prompt.format_messages()], callbacks=callbacks
    #             )
    #         self.langchain_llm.n = old_n
    #     else:
    #         if isinstance(self.llm, BaseLLM):
    #             list_llmresults: list[LLMResult] = run_async_tasks(
    #                 [
    #                     self.llm.agenerate([prompt.format()], callbacks=callbacks)
    #                     for _ in range(n)
    #                 ]
    #             )
    #         else:
    #             list_llmresults: list[LLMResult] = run_async_tasks(
    #                 [
    #                     self.llm.agenerate(
    #                         [prompt.format_messages()], callbacks=callbacks
    #                     )
    #                     for _ in range(n)
    #                 ]
    #             )

    #         # fill results as if the LLM supported multiple completions
    #         generations = [r.generations[0][0] for r in list_llmresults]
    #         llm_output = _compute_token_usage_langchain(list_llmresults)
    #         result = LLMResult(generations=[generations], llm_output=llm_output)

    #     return result

    # def generate(
    #     self,
    #     prompts: list[ChatPromptTemplate],
    #     n: int = 1,
    #     temperature: float = 1e-8,
    #     callbacks: t.Optional[Callbacks] = None,
    # ) -> LLMResult:
    #     # set temperature to 0.2 for multiple completions
    #     temperature = 0.2 if n > 1 else 1e-8
    #     if isBedrock(self.llm) and ("model_kwargs" in self.llm.__dict__):
    #         self.llm.model_kwargs = {"temperature": temperature}
    #     elif isAmazonAPIGateway(self.llm) and ("model_kwargs" in self.llm.__dict__):
    #         self.llm.model_kwargs = {"temperature": temperature}
    #     else:
    #         self.llm.temperature = temperature

    #     if self.llm_supports_completions(self.llm):
    #         return self._generate_multiple_completions(prompts, n, callbacks)
    #     else:  # call generate_completions n times to mimic multiple completions
    #         list_llmresults = run_async_tasks(
    #             [self.generate_completions(prompts, callbacks) for _ in range(n)]
    #         )

    #         # fill results as if the LLM supported multiple completions
    #         generations = []
    #         for i in range(len(prompts)):
    #             completions = []
    #             for result in list_llmresults:
    #                 completions.append(result.generations[i][0])
    #             generations.append(completions)

    #         llm_output = _compute_token_usage_langchain(list_llmresults)
    #         return LLMResult(generations=generations, llm_output=llm_output)
        
    