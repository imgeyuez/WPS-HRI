import requests
import json

def llm_inference(api_key:str, model:str, prompt:str):
    """
    Function which prompts to the LLM and saves the response
    in a variable.
    """

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        data=json.dumps({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2
    })
)

    # Parse response JSON
    if response.status_code == 200:
        response_json = response.json()
        # Extract the model's reply
        answer = response_json['choices'][0]['message']['content']

    else:
        answer = f"Error: {response.status_code}, {response.text}"
        print(f"Error: {response.status_code}, {response.text}")

    return answer
        



# pip install ollama
"""
Previous script to interact with models on ollama
Just keeping it in case. 

import ollama 

def llm_inference(model, prompt, temperature=0.7):
    response = ollama.chat(
        model=model, 
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': temperature}  # Set the temperature
    )

    # return the answer of model
    return response['message']['content']
"""