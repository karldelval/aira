import ollama

# Define your prompt
prompt = "What is the capital of France?"

# Send the prompt to the model and get the response
response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])

# Print the entire response to inspect its structure
print(response)