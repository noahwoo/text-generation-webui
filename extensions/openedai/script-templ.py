import base64
import json
import os
import time
import requests
import yaml
import datetime

def custom_generate_reply(question, original_question, seed, state, eos_token, stopping_strings, is_chat = True):

    # Add a user message
    print(f"custom_generate(question): {question}\n")
    print(f"custom_generate(original_question): {original_question}\n")
    print(f"custom_generate(seed): {seed}\n")
    print(f"custom_generate(state): {state}\n")
    print(f"custom_generate(eos_token): {eos_token}\n")
    print(f"custom_generate(stopping_strings): {stopping_strings}\n")

    cumulative = f'{question}\n'
    for i in range(10):
        cumulative += f"Counting: {i}...\n"
        yield cumulative

    cumulative += f"Done! {str(datetime.datetime.now())}"
    yield cumulative
