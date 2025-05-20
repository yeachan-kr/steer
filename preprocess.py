# prompt for llama2-chat
PROMPT_BEGIN: str = ''
PROMPT_USER: str = '[INST] {input} '
PROMPT_ASSISTANT: str = '[/INST]'  # should not have a space at the end

PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

PROMPT_DICT: dict[str, str] = {
    'prompt_begin': PROMPT_BEGIN,
    'prompt_user': PROMPT_USER,
    'prompt_assistant': PROMPT_ASSISTANT,
    'prompt_input': PROMPT_INPUT,
}

def preprocess(input, eos_token):
    data = [PROMPT_BEGIN]
    for idx, line in enumerate([input]):
        # PROMPT_INPUT
        if idx % 2 == 0:
            data.extend((PROMPT_USER.format(input=line), PROMPT_ASSISTANT))
        else:
            if eos_token == "empty": # inference
                None
            else: # train
                data.extend((line, eos_token))

    return ''.join(data)