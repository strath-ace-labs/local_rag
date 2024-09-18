#from datasets import from_pandas load_dataset
import pandas as pd 
import datasets
import json 

import json 
import argparse 

from langchain.prompts import ChatPromptTemplate

from tqdm.notebook import tqdm

from ragas.llms.huggingface import HuggingfaceLLM
from langchain.prompts import HumanMessagePromptTemplate


def create_prompt_string(prompt_reason="", user_prompt="", definition = "", req= "", system_prompt = "", cot= True, internal_cot = False):
    if cot:
        return f"{user_prompt}{prompt_reason}\n{system_prompt}" +"Let's think step-by-step:"
    elif internal_cot: 
        return  f"{user_prompt}{prompt_reason}\n"
    else: 
        #return f"{user_prompt} {prompt_reason}\n{system_prompt} "
        return f"{user_prompt}{prompt_reason}\n{system_prompt}"


def main(inputs):

    #with open(inputs['input_path'], "r") as f:
    #    data = json.load(f)

    #df = pd.DataFrame(data=data)
    
    df = pd.read_json(inputs['input_path'])
    
    print(df.shape)
    if inputs['one_shot']:
        prompt_reason = """
                        Answer the question using the information from the given context.
                        ________________________
                        Example Question: What is the keyword that best describes the paper's focus in natural language understanding tasks?
                        Example Context: 1. Currently a LEO satellite can only exchange TT&C and other data during its pass over a ground station which can be limited to <10% of the orbit. The SB-SAT system solves this problem providing up to 100% coverage.
                        2. The SB-SAT ARTES 3-4 programme has covered the development of the SB-SAT (Swift Broadband for Satellite) system that provides real time, on demand, persistent broadband ground connectivity for spacecraft in Low Earth Orbits (LEO), using the Inmarsat BGAN system, the Inmarsat-4 satellite constellation and a new spacecraft terminal engineered for the space environment, orbits, altitudes and velocities consistent with LEO spacecraft.
                        3. The system required two developments. Upgrades to the Inmarsat BGAN ground infrastructure and the development of a space qualified SB-SAT terminal, a new class of terminal (based on the Inmarsat Swift Broad Band Aero Class terminals) for seamless use within the upgraded network.
                        4. The terminal provides a persistent data link capable of supporting TT&C and Mission
                        Example Answer: Several key developments were needed for the enhanced communication between a Low Earth Orbit (LEO) spacecraft and its mission control center, primarily facilitated by the SB-SAT system. A crucial component of the SB-SAT system's success was the significant upgrades made to the Inmarsat BGAN ground infrastructure. This allowed to leverage the Inmarsat-4 satellite constellation, part of the Inmarsat BGAN (Broadband Global Area Network) system, to maintain continuous, on-demand broadband connectivity for LEO spacecraft. Another key development was the creation of a new class of terminal specifically designed for use in the space environment. This space-qualified SB-SAT terminal based on the Inmarsat Swift Broad Band Aero Class terminals allows for seamless use within the upgraded network and is engineered for the space environment, orbits, altitudes and velocities consistent with LEO spacecraft. As a result the SB-SAT system provides an increase in ground communication from former 10% to now offering up to 100% coverage for a persistent data link capable of supporting TT&C. 
                        ________________________
                        Example Question: {question}
                        Example Context: {context} 
                        Answer:   
                        """
        
    elif inputs['three_shot']: 
        prompt_reason = """
        Answer the question using the information from the given context.
        ________________________
        Example Question: What role did the collector design optimizations play in confirming the performance of the 300W KU-Band TWT TL 12310C's inner conductor helix connection?
        Example Context: 2.6 C OLLECTOR
        For the Ku-Band 300W Project was planned to use a brazed electrostatic five stage collector to increase the overall efficiency. As such a collector doesn’t exist it has to be designed “right from the scratch”. Therefore with Collect3d (Thales software for collector design) the first model were simulated and calculated. The geometry, outer diameter of 36.5 mm with 4 mm ceramic segments, of the collector has been optimized for the center frequency 11.75 GHz. The edge frequencies of 10.7 GHz and 12.75 GHz have been checked concerning the back streaming of electrons and the efficiency. The final collector model was optimized concerning the efficiency and back streaming at saturation drive of the tube. Furthermore, all drive levels at a frequency of 11.75 GHz were calculated from zero to +10 dB Overdrive and no problems concerning back streaming of electrons could be detected.
        Further operating modes (Overdrive, Saturation, IBO w.c.) are listed in Table 2-1: Breakdown list of the current distribution in the collector.
        Electrode
        Voltage
        Zero Drive
        IBO -12 dB (Qdiss wc)
        Sat drive
        Overdrive +6 dB
        U Helix
        Example Answer: The collector was specifically designed from scratch using Thales' Collect3d software to increase the overall efficiency of the Traveling Wave Tube (TWT). By optimizing the geometry and the outer diameter of the collector to 36.5 mm with 4 mm ceramic segments, the design aimed to maximize efficiency at the center frequency of 11.75 GHz. The design also took into account the edge frequencies of 10.7 GHz and 12.75 GHz, evaluating the collector's performance regarding the back streaming of electrons and efficiency at these frequencies. The optimizations included calculations of all drive levels at the frequency of 11.75 GHz, from zero to +10 dB Overdrive. These calculations indicated no problems concerning back streaming of electrons, which underscores the collector design's effectiveness in handling a wide range of operational conditions without compromising performance.
        ________________________
        Example Question: What is the keyword that best describes the paper's focus in natural language understanding tasks?
        Example Context: 1. Currently a LEO satellite can only exchange TT&C and other data during its pass over a ground station which can be limited to <10% of the orbit. The SB-SAT system solves this problem providing up to 100% coverage.
        2. The SB-SAT ARTES 3-4 programme has covered the development of the SB-SAT (Swift Broadband for Satellite) system that provides real time, on demand, persistent broadband ground connectivity for spacecraft in Low Earth Orbits (LEO), using the Inmarsat BGAN system, the Inmarsat-4 satellite constellation and a new spacecraft terminal engineered for the space environment, orbits, altitudes and velocities consistent with LEO spacecraft.
        3. The system required two developments. Upgrades to the Inmarsat BGAN ground infrastructure and the development of a space qualified SB-SAT terminal, a new class of terminal (based on the Inmarsat Swift Broad Band Aero Class terminals) for seamless use within the upgraded network.
        4. The terminal provides a persistent data link capable of supporting TT&C and Mission
        Example Answer: Several key developments were needed for the enhanced communication between a Low Earth Orbit (LEO) spacecraft and its mission control center, primarily facilitated by the SB-SAT system. A crucial component of the SB-SAT system's success was the significant upgrades made to the Inmarsat BGAN ground infrastructure. This allowed to leverage the Inmarsat-4 satellite constellation, part of the Inmarsat BGAN (Broadband Global Area Network) system, to maintain continuous, on-demand broadband connectivity for LEO spacecraft. Another key development was the creation of a new class of terminal specifically designed for use in the space environment. This space-qualified SB-SAT terminal based on the Inmarsat Swift Broad Band Aero Class terminals allows for seamless use within the upgraded network and is engineered for the space environment, orbits, altitudes and velocities consistent with LEO spacecraft. As a result the SB-SAT system provides an increase in ground communication from former 10% to now offering up to 100% coverage for a persistent data link capable of supporting TT&C. 
        ________________________
        Example Question: Question: \"Can you identify the three types of interference that need to be factored in when examining the downlink scenario involving CGC user terminals and MSS satellites?\"
        Example Context: Project: ESA_1-5800\nContract Ref:\n22036/08/NL/AD\nRef.\nIssue\nRev.\nDate\nPage\nESA_1-5800_ASTD_TN8_2\nFinal rev1\n0\n12/12/2011\n61 of 79\nFile name: esa_1-5800_astd_finrep_rev1_20111212.doc\nSome results of sharing scenarios for the MSS uplink band\n(1) Co-frequency interference from a CGC base station to an MSS\nsatellite receiver\nThere is the potential for interference from CGC user terminals to an MSS\nsatellite receiver. As for the downlink situation, it is necessary to consider\nthree interference mechanisms: co-frequency interference from CGC user\nterminals to the MSS satellite; overload of the MSS satellite receiver due to\ninterference anywhere in the uplink band; and interference from the\nunwanted emissions of CGC UTs into the MSS satellite. For each case, the\nnumber of CGC UTs at which the criterion is just met can be determined.\n(2) Co-frequency interference from CGC user terminals to an MSS\nsatellite\nFor the case of CGC operations on the same frequencies as MSS operations,\nit is assumed that the area in which CGC is deployed is outside of the\nsatellite beam. Hence the CGC interference to the MSS satellite will be\nreceived with an antenna gain lower than the peak value. Figure 4-2 shows\nthe gain contours for an example Inmarsat-4 spot beam.\n0.00\n-20.00\n-15.00\n-10.00\n-8.00\n-6.00\n-4.00\n-2.00\nFigure 4-2: Example spot beam contours with example CGC deployment area overlaid\nIf we consider CGC to be deployed in the blue circle, it can be seen that in\nthis example the antenna discrimination ranges from about 3 dB to about 15\ndB. In principle, it would be necessary to determine the average antenna\ndiscrimination existing for all UTs deployed in the circle by integrating across\nthe circle. For the purpose of this analysis and for simplicity, we assume an\naverage discrimination of 9 dB.
        Example Answer: 1. Co-frequency interference from CGC user terminals to the MSS satellite;
        2. Interference from the unwanted emissions of CGC UTs into the MSS satellite; 
        3. Overload of the MSS satellite receiver due to interference anywhere in the uplink band
        ________________________
        Example Question: {question}
        Example Context: {context} 
        Answer:  
        """
    else:
        prompt_reason = """\
        Answer the question using the information from the given context. 
        Question: {question}
        Context: {context}
        Answer: 
        """

    
    model_name = inputs['model_name']#"microsoft/phi-2" # "openchat/openchat_3.5"#  
    model = HuggingfaceLLM(model_name)

    if "openchat" in model.huggingface_llm.config._name_or_path:

        system_prompt = "GPT4 Correct Assistant: "
        user_prompt = "GPT4 Correct User: "
    
    else: 
        print("else")
        system_prompt = ""
        user_prompt = ""

    #prompt = create_prompt_string(prompt_reason=prompt_reason, user_prompt=user_prompt, system_prompt=system_prompt, cot=inputs['cot'])
    prompt = prompt_reason

    ANSWER_FORMULATE = HumanMessagePromptTemplate.from_template(prompt)
    #     """\
    #     Answer the question using the information from the given context. 
    #     Question: {question}
    #     Context: {context}
    #     Answer: 
    # """  # noqa: E501
    

    # ANSWER_FORMULATE = HumanMessagePromptTemplate.from_template(
    #     """\
    #     Answer the question.  
    #     Question: {question}
    #     Answer: 
    # """  # noqa: E501
    # )
    #   Context: {context}



    answers = []
    for row in tqdm(df.index):

        #print(row)
        question = df.loc[row, "question"]
        context = df.loc[row, "context"]
        context = df.loc[row, "text_chunk"]

        human_prompt = ANSWER_FORMULATE.format(question=question, context=context)
        prompt = ChatPromptTemplate.from_messages([human_prompt])

        if "phi" in model.huggingface_llm.config._name_or_path:
            ## print("phi")
            results = model.generate([prompt], max_new_tokens= 512, repetition_penalty=1.1)
        
        else: 
            results = model.generate([prompt], max_new_tokens= 1000)

        output = results.generations[0][0].text.strip()
        answers.append({"prompt":human_prompt,'output':output,})

    df = pd.DataFrame(answers)
    # df.to_json(f"{model_name}_cot_{str(inputs['cot'])}_answers.json", orient='records', indent=4)
    df.to_json(inputs['output_path'], orient='records', indent=4)

    #df[f'{model_name}_answer'] = answers

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process the config file path.')
    parser.add_argument('config_path', help='Path to the config.json file')
    
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config= json.load(file)
    inputs = config 
    main(inputs)