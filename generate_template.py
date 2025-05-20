# GPT
# CAUSION의 1번째 문장은 답만 생성하는 케이스를 막기 위한 조치. 무조건 배경 지식 + 정답 지식 + 그에 얼라인된 정답 + 각 스텝에 해당되는 confidence score를 응답해야 한다. 
system_gpt = """You are a helpful assistant in extracting the background knowledge to solve the given question and the answer knowledge related to the correct answer.

## Guidelines
1. Your task is to extract the background knowledge to easily solve the given question and the answer knowledge related to the correct answer.
2. Specifically, the extracted background knowledge should be relevant to the given input and answer choices.
3. And, extracted answer knowledge should be relevant to only correct answer.
4. Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer.
5. Note: The confidence indicates how likely you think your answer is true.
6. Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY a complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
- Final_Answer: [ONLY the answer option from A to H; not a complete sentence]
- Overall Confidence (0-100): [Your confidence value]%
""

## CAUSION
- You MUST NOT respond any of the given answer choices as the answer.
- You should correctly generate the knowledge.
"""


system_gpt_arith = """You are a helpful assistant in extracting the intermediate reasoning to solve the given question and the answer reasoning synthesized from intermediate reasoning.

## Guidelines
1. Your task is to extract the intermediate reasoning and the answer reasoning in order to easily solve the given question.
2. Specifically, the extracted intermediate reasoning is defined as the step by step process used to solve the question, and the extracted answer reasoning is defined as the final reasoning synthesized from intermediate reasoning.
3. Given all of the above, read the question, think step by step, break down the question into N steps of intermediate reasoning, give your confidence in each step, and then derive your final answer reasoning and your confidence in this answer.
4. Note: The confidence indicates how likely you think your reasoning is correct.
5. Use the following format to answer:
""
Intermediate_Reasoning_1: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Intermediate_Reasoning_2: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
...
Intermediate_Reasoning_N: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Answer_Reasoning: [Your answer reasoning; ONLY a complete answer reasoning sentence; not the answer type], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
- Final_Answer: [ONLY your answer as single numeric; not a complete sentence]
- Overall Confidence (0-100): [Your confidence value]%
""

## CAUSION 
1. Correctly generate the reasoning and answer.
2. Strictly follow the answer format mentioned.
3. Do not overestimate your confidence.
"""


stqa_gpt = """
# Question
Input: {goal}
A. True
B. False
"""

csqa_gpt = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
"""

piqa_gpt = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
"""

qasc_gpt = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
F. {sol6}
G. {sol7}
H. {sol8}
"""

obqa_gpt = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
"""

arc_h_gpt = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
"""

wngr_gpt = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
"""

siqa_gpt = """
# Question
Context: {context}
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
"""

boolq_gpt = """
# Question
Passage: {context}
Input: {goal}
A. True
B. False
"""

gsm8k_gpt = """
# Question
Input: {goal}
"""


##### MISTRAL #####
obqa_user_mistral = """You are a helpful chatbot.
Your task is to extract the background knowledge to easily solve the given question and the answer knowledge related to the correct answer.
# The extracted background knowledge should be relevant to the given input and answer choices.
# Extracted answer knowledge should be relevant to only correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}

Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note that The confidence indicates how likely you think your answer is true.

# Please use the following format to answer
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to D; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

arc_user_mistral = """You are a helpful chatbot.
Your task is to extract the background knowledge to easily solve the given question and the answer knowledge related to the correct answer.
# The extracted background knowledge should be relevant to the given input and answer choices.
# Extracted answer knowledge should be relevant to only correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}

Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note that The confidence indicates how likely you think your answer is true.

# Please use the following format to answer
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to D; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

csqa_user_mistral = """You are a helpful chatbot.
Your task is to extract the background knowledge to easily solve the given question and the answer knowledge related to the correct answer.
# The extracted background knowledge should be relevant to the given input and answer choices.
# Extracted answer knowledge should be relevant to only correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}

Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note that The confidence indicates how likely you think your answer is true.

# Please use the following format to answer
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to E; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

qasc_user_mistral = """You are a helpful chatbot.
Your task is to extract the background knowledge to easily solve the given question and the answer knowledge related to the correct answer.
# The extracted background knowledge should be relevant to the given input and answer choices.
# Extracted answer knowledge should be relevant to only correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
F. {sol6}
G. {sol7}
H. {sol8}

Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note that The confidence indicates how likely you think your answer is true.

# Please use the following format to answer
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to H; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

piqa_user_mistral = """You are a helpful chatbot.
Your task is to extract the background knowledge to easily solve the given question and the answer knowledge related to the correct answer.
# The extracted background knowledge should be relevant to the given input and answer choices.
# Extracted answer knowledge should be relevant to only correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. {sol1}
B. {sol2}

Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note that The confidence indicates how likely you think your answer is true.

# Please use the following format to answer
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to B; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""


siqa_user_mistral = """You are a helpful chatbot.
Your task is to extract the background knowledge to easily solve the given question and the answer knowledge related to the correct answer.
# The extracted background knowledge should be relevant to the given input and answer choices.
# Extracted answer knowledge should be relevant to only correct answer.
You should correctly generate the knowledge.

### Question
Context: {context}
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}

Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note that The confidence indicates how likely you think your answer is true.

# Please use the following format to answer
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to C; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

wngr_user_mistral = """You are a helpful chatbot.
Your task is to extract the background knowledge to easily solve the given question and the answer knowledge related to the correct answer.
# The extracted background knowledge should be relevant to the given input and answer choices.
# Extracted answer knowledge should be relevant to only correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. {sol1}
B. {sol2}

Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note that The confidence indicates how likely you think your answer is true.

# Please use the following format to answer
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to B; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

stqa_user_mistral = """You are a helpful chatbot.
Your task is to extract the background knowledge to easily solve the given question and the answer knowledge related to the correct answer.
# The extracted background knowledge should be relevant to the given input and answer choices.
# Extracted answer knowledge should be relevant to only correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. True
B. False

Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note that The confidence indicates how likely you think your answer is true.

# Please use the following format to answer
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to B; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""




######## Llama 2 template #############
######################### system template#########################
# system_prompt_2 = """You are a helpful assistant in extracting the background knowledge to solve the given question and the answer knowledge related to the correct answer."""

system_prompt_2 = """You are a helpful assistant in extracting the background knowledge to solve the given question and the answer knowledge related to the correct answer.

# Guidelines
""
- Your task is to extract the background knowledge to easily solve the given question and the answer knowledge related to the correct answer.
- Specifically, the extracted background knowledge should be relevant to the given input and answer choices.
- And, extracted answer knowledge should be relevant to only correct answer.
""

# CAUSION
""
- You MUST NOT respond any of the given answer choices as the answer.
- Correctly generate the knowledge.
- Strictly follow the answer format mentioned.
""
"""
######################### task user template#########################
wngr_user_2 = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}

Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

# Please use the following format to answer.
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to B; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

Assistant:"""

siqa_user_2 = """
# Question
Context: {context}
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}

Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to B; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
Generate all requested items without omission, strictly following the given format.
Do not generate overly verbose knowledge.

Assistant: The Response is ("""

obqa_user_2 = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}

Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to B; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
Generate all requested items without omission, strictly following the given format.
Do not generate overly verbose knowledge.

Assistant: The Response is ("""

arc_user_2 = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}

Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to B; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
Generate all requested items without omission, strictly following the given format.
Do not generate overly verbose knowledge.

Assistant: The Response is ("""

piqa_user_2 = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}

Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to B; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
Generate all requested items without omission, strictly following the given format.
Do not generate overly verbose knowledge.

Assistant: The Response is ("""

# STQA
stqa_user_2 = """
# Question
Input: {goal}
A. True
B. False

Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to B; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
Generate all requested items without omission, strictly following the given format.
Do not generate overly verbose knowledge.

Assistant: The Response is ("""

# CSQA
csqa_user_2 = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}

Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to B; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
Generate all requested items without omission, strictly following the given format.
Do not generate overly verbose knowledge.

Assistant: The Response is ("""


qasc_user_2 = """
### Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
F. {sol6}
G. {sol7}
H. {sol8}

Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY one complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY one complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY one answer option from A to B; not a answer content]
Overall Confidence (0-100): [Your confidence value]%
""
Generate all requested items without omission, strictly following the given format.
Do not generate overly verbose knowledge.

Assistant: The Response is ("""

##### Llama-3.2-3B-Instruct template ###############
######################### system template#########################
system_prompt = """You are a helpful assistant in extracting the background knowledge to solve the given question and the answer knowledge related to the correct answer.

# Guidelines
""
- Your task is to extract the background knowledge to easily solve the given question and the answer knowledge related to the correct answer.
- Specifically, the extracted background knowledge should be relevant to the given input and answer choices.
- And, extracted answer knowledge should be relevant to only correct answer.
""

# CAUSION
""
- You MUST NOT respond any of the given answer choices as the answer.
- Correctly generate the knowledge.
- Strictly follow the answer format mentioned.
- Do not overestimate your confidence.
""
"""

system_prompt_arith = """You are a helpful assistant in extracting the intermediate reasoning to solve the given question and the answer reasoning synthesized from intermediate reasoning.

# Guidelines
""
- Your task is to extract the intermediate reasoning and the answer reasoning in order to easily solve the given question.
- Specifically, the extracted intermediate reasoning is defined as the step by step process used to solve the question.
- And, the extracted answer reasoning is defined as the final reasoning synthesized from intermediate reasoning.
""

# CAUSION
""
- Correctly generate the reasoning and answer.
- Strictly follow the answer format mentioned.
- Do not overestimate your confidence.
""
"""
######################### task user template#########################
# length 문제가 있어, 4개까지 허용.
aqua_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
Human: Given all of the above, read the question, think step by step, break down the question into 4 steps of intermediate reasoning, give your confidence in each step, and then derive your final answer reasoning and your confidence in this answer. Note: The confidence indicates how likely you think your reasoning is correct.

Use the following format to answer:
""
Intermediate_Reasoning_1: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Intermediate_Reasoning_2: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Intermediate_Reasoning_3: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Intermediate_Reasoning_4: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Answer_Reasoning: [Your answer reasoning; ONLY a complete answer reasoning sentence; not the answer type], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to E; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

gsm8k_user = """
# Question
Input: {goal}
Human: Given all of the above, read the question, think step by step, break down the question into N steps of intermediate reasoning, give your confidence in each step, and then derive your final answer reasoning and your confidence in this answer. Note: The confidence indicates how likely you think your reasoning is correct.

Use the following format to answer:
""
Intermediate_Reasoning_1: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Intermediate_Reasoning_2: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
...
Intermediate_Reasoning_N: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Answer_Reasoning: [Your answer reasoning; ONLY a complete answer reasoning sentence; not the answer type], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Final_Answer: [ONLY your answer as single numeric; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""


# Boolq
boolq_user = """
# Question
Passage: {context}
Input: {goal}
A. True
B. False
Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY a complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""


acl_arc_user = """
# Question
Classify the given input into the appropriate scientific citation category (from A to F).
Input: {goal}
A. background
B. motivation
C. compareorcontrast
D. uses
E. extends
F. future
Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY a complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to F; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

chemprot_user = """
# Question
Classify the given input into the appropriate chemical-protein-disease annotation category (from A to M).
Input: {goal}
A. INHIBITOR
B. SUBSTRATE
C. INDIRECT-UPREGULATOR
D. INDIRECT-DOWNREGULATOR
E. ACTIVATOR
F. ANTAGONIST
G. PRODUCT-OF
H. AGONIST
I. DOWNREGULATOR
J. UPREGULATOR
K. AGONIST-ACTIVATOR
L. SUBSTRATE_PRODUCT-OF
M. AGONIST-INHIBITOR
Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY a complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to M; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""



wngr_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY a complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

siqa_user = """
# Question
Context: {context}
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY a complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to C; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

obqa_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY a complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

arc_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY a complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

piqa_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY a complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

# STQA
stqa_user = """
# Question
Input: {goal}
A. True
B. False
Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY a complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

# CSQA
csqa_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY a complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to E; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""


qasc_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
F. {sol6}
G. {sol7}
H. {sol8}
Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY a complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to H; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""