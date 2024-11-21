import json
import random
import time
from openai import OpenAI  # Import OpenAI from OpenRouter

from utils import *

# Initialize OpenAI client for OpenRouter endpoint
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openai_api_key,
)

def temp_sleep(seconds=0.1):
    time.sleep(seconds)

def ChatGPT_single_request(prompt): 
    temp_sleep()

    # Using OpenRouter API client instead of openai directly
    completion = client.chat.completions.create(
        model="openai/gpt-3.5-turbo", 
        messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt): 
    temp_sleep()

    try: 
        completion = client.chat.completions.create(
            model="openai/gpt-4", 
            messages=[{"role": "user", "content": prompt}]
        )
        return completion["choices"][0]["message"]["content"]
    except Exception as e: 
        print(f"ChatGPT ERROR: {e}")
        return "ChatGPT ERROR"


def ChatGPT_request(prompt): 
    temp_sleep()

    try: 
        completion = client.chat.completions.create(
            model="openai/gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}]
        )
        return completion["choices"][0]["message"]["content"]
    except Exception as e: 
        print(f"ChatGPT ERROR: {e}")
        return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt, 
                               example_output,
                               special_instruction,
                               repeat=3,
                               fail_safe_response="error",
                               func_validate=None,
                               func_clean_up=None,
                               verbose=False): 
    prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    if verbose: 
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat): 
        try: 
            curr_gpt_response = GPT4_request(prompt).strip()
            end_index = curr_gpt_response.rfind('}') + 1
            curr_gpt_response = curr_gpt_response[:end_index]
            curr_gpt_response = json.loads(curr_gpt_response)["output"]
            
            if func_validate(curr_gpt_response, prompt=prompt): 
                return func_clean_up(curr_gpt_response, prompt=prompt)
            
            if verbose: 
                print(f"---- repeat count: {i}\n{curr_gpt_response}\n~~~~")

        except Exception as e: 
            print(f"Error: {e}")
            pass

    return False


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
    prompt = '"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    if verbose: 
        print("CHAT GPT PROMPT")
        print(prompt)

    for i in range(repeat): 
        try: 
            curr_gpt_response = ChatGPT_request(prompt).strip()
            end_index = curr_gpt_response.rfind('}') + 1
            curr_gpt_response = curr_gpt_response[:end_index]
            curr_gpt_response = json.loads(curr_gpt_response)["output"]
            
            if func_validate(curr_gpt_response, prompt=prompt): 
                return func_clean_up(curr_gpt_response, prompt=prompt)
            
            if verbose: 
                print(f"---- repeat count: {i}\n{curr_gpt_response}\n~~~~")

        except Exception as e: 
            print(f"Error: {e}")
            pass

    return False


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
    if verbose: 
        print(prompt)

    for i in range(repeat): 
        curr_gpt_response = GPT_request(prompt, gpt_parameter)
        if func_validate(curr_gpt_response, prompt=prompt): 
            return func_clean_up(curr_gpt_response, prompt=prompt)
        if verbose: 
            print(f"---- repeat count: {i}\n{curr_gpt_response}\n~~~~")
    return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter): 
    temp_sleep()
    try: 
        response = client.completion.create(
            model=gpt_parameter["engine"],
            prompt=prompt,
            temperature=gpt_parameter["temperature"],
            max_tokens=gpt_parameter["max_tokens"],
            top_p=gpt_parameter["top_p"],
            frequency_penalty=gpt_parameter["frequency_penalty"],
            presence_penalty=gpt_parameter["presence_penalty"],
            stream=gpt_parameter["stream"],
            stop=gpt_parameter["stop"],
        )
        return response.choices[0].text
    except Exception as e: 
        print(f"TOKEN LIMIT EXCEEDED: {e}")
        return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file): 
    if type(curr_input) == type("string"): 
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    with open(prompt_lib_file, "r") as f:
        prompt = f.read()

    for count, i in enumerate(curr_input):   
        prompt = prompt.replace(f"!<INPUT {count}>!", i)

    if "<commentblockmarker>###</commentblockmarker>" in prompt: 
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()


# Example usage
if __name__ == '__main__':
    gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50, 
                     "temperature": 0, "top_p": 1, "stream": False,
                     "frequency_penalty": 0, "presence_penalty": 0, 
                     "stop": ['"']}
    curr_input = ["driving to a friend's house"]
    prompt_lib_file = "prompt_template/test_prompt_July5.txt"
    prompt = generate_prompt(curr_input, prompt_lib_file)

    def __func_validate(gpt_response): 
        return len(gpt_response.strip()) > 1 and len(gpt_response.strip().split(" ")) > 1

    def __func_clean_up(gpt_response):
        return gpt_response.strip()

    output = safe_generate_response(prompt, 
                                     gpt_parameter,
                                     5,
                                     "rest",
                                     __func_validate,
                                     __func_clean_up,
                                     True)

    print(output)
