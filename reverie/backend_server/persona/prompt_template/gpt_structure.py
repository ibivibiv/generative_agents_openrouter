"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure_openrouter.py
Description: Wrapper functions for calling OpenRouter's OpenAI-compatible APIs.
"""
import json
import random
import time

from openrouter import OpenRouter
from utils import *

# Initialize OpenRouter client
client = OpenRouter(api_key='your_openrouter_api_key')  # Replace with your API key

def temp_sleep(seconds=0.1):
    time.sleep(seconds)

def ChatGPT_single_request(prompt):
    temp_sleep()

    completion = client.chat_completion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]

# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt):
    """
    Given a prompt, make a request to OpenRouter server and returns the response.
    ARGS:
        prompt: a str prompt
    RETURNS:
        a str of GPT-4's response.
    """
    temp_sleep()

    try:
        completion = client.chat_completion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"ChatGPT ERROR: {e}")
        return "ChatGPT ERROR"

def ChatGPT_request(prompt):
    """
    Given a prompt, make a request to OpenRouter server and returns the response.
    ARGS:
        prompt: a str prompt
    RETURNS:
        a str of GPT-3.5-turbo's response.
    """
    try:
        completion = client.chat_completion.create(
            model="gpt-3.5-turbo",
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
    prompt = 'GPT-4 Prompt:\n"""\n' + prompt + '\n"""\n'
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
                print("---- repeat count: \n", i, curr_gpt_response)
                print(curr_gpt_response)
                print("~~~~")
        except Exception as e:
            print(f"Error during response generation: {e}")

    return fail_safe_response

# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenRouter
    server and returns the response.
    ARGS:
        prompt: a str prompt
        gpt_parameter: a python dictionary with the keys indicating the names of  
                    the parameter and the values indicating the parameter 
                    values.   
    RETURNS: 
        a str of GPT-3's response. 
    """
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
            stop=gpt_parameter["stop"]
        )
        return response.choices[0].text
    except Exception as e:
        print(f"TOKEN LIMIT EXCEEDED: {e}")
        return "TOKEN LIMIT EXCEEDED"

# Example Main Function
if __name__ == '__main__':
    gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50,
                     "temperature": 0, "top_p": 1, "stream": False,
                     "frequency_penalty": 0, "presence_penalty": 0,
                     "stop": ['"']}
    curr_input = ["driving to a friend's house"]
    prompt_lib_file = "prompt_template/test_prompt_July5.txt"
    prompt = generate_prompt(curr_input, prompt_lib_file)

    def __func_validate(gpt_response):
        if len(gpt_response.strip()) <= 1:
            return False
        if len(gpt_response.strip().split(" ")) > 1:
            return False
        return True

    def __func_clean_up(gpt_response):
        cleaned_response = gpt_response.strip()
        return cleaned_response

    output = safe_generate_response(prompt,
                                     gpt_parameter,
                                     5,
                                     "rest",
                                     __func_validate,
                                     __func_clean_up,
                                     True)

    print(output)
