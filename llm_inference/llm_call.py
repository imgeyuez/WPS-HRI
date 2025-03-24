from together import Together
import time

def llm_inference(api_key:str, model:str, prompt:str):
    """
    Function which prompts to the LLM and saves the response
    in a variable.

    Input
    api_key (str)   : String of the API-key
    model (str)     : Name of the model
    prompt (str)    : Prompt to give the model

    Output
    answer (str)    : Response of the LLM
    """
    
    # Initialize the Together client with the correct argument name
    client = Together(api_key=api_key)

    try:
        # Send a message to the Llama model
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        # save the response from the model
        answer = response.choices[0].message.content

        print(f"{model} generated answer")

        if model == "deepseek-ai/DeepSeek-R1":
            # wait 200seconds bc of request limits 
            print("Waiting for 200 seconds to avoid request limit...")
            time.sleep(200)

        return answer

    except Exception as e:
        print(f"Error: {e}")
        print("Retrying after 200 seconds...")
        time.sleep(200)


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

# inferencee using openrouter
'''
def llm_inference(api_key:str, model:str, prompt:str):
    """
    Function which prompts to the LLM and saves the response
    in a variable.

    Input
    api_key (str)   : String of the API-key
    model (str)     : Name of the model
    prompt (str)    : Prompt to give the model

    Output
    answer (str)    : Response of the LLM
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
        try:
            response_json = response.json()
            print("Full Response JSON:", json.dumps(response_json, indent=4))  # Print the full response for debugging

            # Check if there's an error in the response
            if "metadata" in response_json and "raw" in response_json["metadata"]:
                error_message = response_json["metadata"]["raw"]
                print("Error in API response:", error_message)
                answer = f"Error: {error_message}"
            else:
                # Try to extract the model's reply if no error in the response
                answer = response_json['choices'][0]['message']['content']
                print("Answer:", answer)

        except KeyError as e:
            print(f"Error extracting data: Missing key {e}")
            answer = "Error: Missing expected data in response."

        except json.JSONDecodeError:
            print("Error decoding JSON from response.")
            answer = "Error: Invalid JSON response."

    else:
        answer = f"Error: {response.status_code}, {response.text}"
        print(f"Error: {response.status_code}, {response.text}")

    return answer

'''