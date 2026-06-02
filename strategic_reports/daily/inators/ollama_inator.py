import os

def get_ollama_client(
    ollama_api_key = os.environ.get('OLLAMA_API_KEY'),
):
    from ollama import Client
    client = Client(
        host="https://ollama.com",
        headers={'Authorization': 'Bearer ' + ollama_api_key}
    )
    return client

def run_ollama_generate_on_client(
    prompt,
    model = 'gpt-oss:120b',
    client = None,
    ollama_api_key = os.environ.get('OLLAMA_API_KEY'),
):
    if client == None:
        client = get_ollama_client(ollama_api_key = ollama_api_key)

    response = client.generate(
        model = model,
        prompt = prompt,
    )
    return response
