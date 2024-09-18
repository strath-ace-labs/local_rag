from langchain.prompts import HumanMessagePromptTemplate

SEED_QUESTION = HumanMessagePromptTemplate.from_template(
#     """\
# Your task is to formulate a question from given context satisfying the rules given below:
#     1.The question should make sense to humans even when read without the given context.
#     2.The question should be fully answered from the given context.
#     3.The question should be framed from a part of context that contains important information. It can also be from tables,code,etc.
#     4.The answer to the question should not contain any links.
#     5.The question should be of moderate difficulty.
#     6.The question must be reasonable and must be understood and responded by humans.
#     7.Do no use phrases like 'provided context',etc in the question
#     8.Avoid framing question using word "and" that can be decomposed into more than one question.
#     9.The question should not contain more than 10 words, make of use of abbreviation wherever possible.
    
# context:{context}
#"""\
#    Given the following context: {context}\n\n----------------\nCreate a question from the context above, the question are asked by a curious user. The question should abide by the following rules and instructions: \n The question has to be strictly based on the context provided above and be self-contained and independent.
#"""  # noqa: E501
"""\
    Create a question from the given context, the question are asked by a curious user. The question should abide by the following rules and instructions: 
    The question has to be strictly based on the context provided above and be self-contained and independent. This means that the question should include all the necessary information and context to be understood and answered without requiring additional external information or assumptions.
    Focus on the Specific Problem or Question specified in the context without tying to a project name or paper title. 
    Example context: "In the 16th century, the Renaissance period saw a significant evolution in art and culture, particularly in Europe. This era was marked by a renewed interest in classical philosophy, literature, and art."
    Question: "How did the Renaissance period's renewed interest in classical philosophy influence European culture in the 16th century?"

    Example context: "{context}"
"""  # noqa: E501

)


REASONING_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
You are a prompt rewriter. You will be provided with a question and a long context.Your task to is to complicate the given question to improve the difficulty of answering. 
You should do complicate the question by rewriting question into a multi-hop reasoning question based on the provided context. The question should require the reader to make multiple logical connections or inferences using the information available in given context. 
Here are some strategies to create multi-hop questions:

   - Bridge related entities: Identify information that relates specific entities and frame question that can be answered only by analysing information of both entities.
   
   - Use Pronouns: identify (he, she, it, they) that refer to same entity or concepts in the context, and ask questions that would require the reader to figure out what pronouns refer to.

   - Refer to Specific Details: Mention specific details or facts from different parts of the context including tables, code, etc and ask how they are related.

   - Pose Hypothetical Scenarios: Present a hypothetical situation or scenario that requires combining different elements from the context to arrive at an answer.

Rules to follow when rewriting question:
1. Ensure that the rewritten question can be answered entirely from the information present in the contexts.
2. Do not frame questions that contains more than 15 words. Use abbreviation wherever possible.
3. Make sure the question is clear and unambiguous. 
4. phrases like 'based on the provided context','according to the context',etc are not allowed to appear in the question.

question: {question}
CONTEXTS:
{context}

Multi-hop Reasoning Question:
"""  # noqa: E501
)

MULTICONTEXT_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
You are a prompt rewriter. You will be provided with a question and two set of contexts namely context1 and context2. 
Your task is to complicate the given question in a way that answering it requires information derived from both context1 and context2. 
Follow the rules given below while rewriting the question.
    1. The rewritten question should not be very long. Use abbreviation wherever possible.
    2. The rewritten question must be reasonable and must be understood and responded by humans.
    3. The rewritten question must be fully answerable from information present in context1 and context2. 
    4. Read and understand both contexts and rewrite the question so that answering requires insight from both context1 and context2.
    5. phrases like 'based on the provided context','according to the context?',etc are not allowed to appear in the question.

question:\n{question}
context1:\n{context1}
context2:\n{context2}
"""  # noqa: E501
)


CONDITIONAL_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Rewrite the provided question to increase its complexity by introducing a conditional element.
The goal is to make the question more intricate by incorporating a scenario or condition that affects the context of the question.
Follow the rules given below while rewriting the question.
    1. The rewritten question should not be longer than 25 words. Use abbreviation wherever possible.
    2. The rewritten question must be reasonable and must be understood and responded by humans.
    3. The rewritten question must be fully answerable from information present context.
    4. phrases like 'provided context','according to the context?',etc are not allowed to appear in the question.
for example,
question: What are the general principles for designing prompts in LLMs?
Rewritten Question:how to apply prompt designing principles to improve LLMs performance in reasoning tasks

question:{question}
context:\n{context}
Rewritten Question
"""  # noqa: E501
)


COMPRESS_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Rewrite the following question to make it more indirect and shorter while retaining the essence of the original question. The goal is to create a question that conveys the same meaning but in a less direct manner.
The rewritten question should be shorter so use abbreviation wherever possible.
Original Question:
{question}

Indirectly Rewritten Question:
"""  # noqa: E501
)


CONVERSATION_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Reformat the provided question into two separate questions as if it were to be part of a conversation. Each question should focus on a specific aspect or subtopic related to the original question.
question: What are the advantages and disadvantages of remote work?
Reformatted Questions for Conversation: What are the benefits of remote work?\nOn the flip side, what challenges are encountered when working remotely?
question:{question}

Reformatted Questions for Conversation:
"""  # noqa: E501
)

SCORE_CONTEXT = HumanMessagePromptTemplate.from_template(
    """Evaluate the provided context and assign a numerical score between 0 and 10 based on the following criteria:
1. Award a high score to context that thoroughly delves into and explains concepts.
2. Assign a lower score to context that contains excessive references, acknowledgments, external links, personal information, or other non-essential elements.
Output the score only.
Context:
{context}
Score:
"""  # noqa: E501
)

FILTER_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Determine if the given question is self-contained and independent and can be clearly understood without assumptions. Assume that the answerer though has a wide general knowledge. Specify the first reason and your verdict in valid json format.
___________________________
Example Question: What is the keyword that best describes the paper's focus in natural language understanding tasks?
Result: {{"reason":"The question does not specify which paper is being referred to, leading to ambiguity and the need for assumptions or external knowledge.", "verdict": "No"}}
___________________________
Example Question:  How do the key challenges and enablers of combining Transmit and Receive X-band Active Antennas address the issues of thermal dissipation and feedback/NPR, and what are the benefits of this configuration in terms of mass efficiency, dual use product, and receive performance optimization? 
Result: {{"reason":"The question is self-contained and independent as it provides specific technical details (Transmit and Receive X-band Active Antennas, thermal dissipation, feedback/NPR) and asks for an explanation of the challenges, enablers, and benefits in a clear context. The question is technical and assumes the answerer has specific knowledge in this field, but it does not rely on external information or specific project details.", "verdict": "Yes"}}
___________________________
Example Question:  What do the primary goals of the UCSM project entail, and how does it tackle the hurdles within the SATCOM sector? 
Result: {{"reason": "Despite using specific acronyms like UCSM and SATCOM, the question is deemed clear and self-sufficient within contexts where such projects and sectors are relevant and known to the audience. It directly asks for the goals and solutions related to recognized challenges, making it accessible to those familiar with the topic.", "verdict": "Yes"}}
___________________________
Example Question: {question}
"""  # noqa: E501
)
##{"reason":"The question assumes knowledge about the UCSM project and specific acronyms like SATCOM, which may not be universally known or understood without additional context.", "verdict": "No"}
## {"reason": "Despite using specific acronyms like UCSM and SATCOM, the question is deemed clear and self-sufficient within contexts where such projects and sectors are relevant and known to the audience. It directly asks for the goals and solutions related to recognized challenges, making it accessible to those familiar with the topic.", "verdict": "Yes"}
## What are the initial managerial deliverables that have been completed for the project, and what is the primary focus of the first Work Package? 
## How do the key challenges and enablers of combining Transmit and Receive X-band Active Antennas address the issues of thermal dissipation and feedback/NPR, and what are the benefits of this configuration in terms of mass efficiency, dual use product, and receive performance optimization? 
## 'What do the primary goals of the UCSM project entail, and how does it tackle the hurdles within the SATCOM sector?'
## "The question assumes knowledge about the UCSM project and specific acronyms like SATCOM, which may not be universally known or understood without additional context."
ANSWER_FORMULATE = HumanMessagePromptTemplate.from_template(
    """\
Answer the question using the information from the given context. 
question:{question}
context:{context}
answer:
"""  # noqa: E501
)

CONTEXT_FORMULATE = HumanMessagePromptTemplate.from_template(
    """Please extract relevant sentences from the provided context that can potentially help answer the following question. While extracting candidate sentences you're not allowed to make any changes to sentences from given context.

question:{question}
context:\n{context}
candidate sentences:\n
"""  # noqa: E501
)
