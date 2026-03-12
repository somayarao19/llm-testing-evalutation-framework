import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load small model (good for laptop testing)
model_name = "distilgpt2"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Model loaded successfully!")

# Read prompts
with open("prompts/factual_prompts.txt", "r") as f:
    prompts = f.readlines()

for prompt in prompts:

    prompt = prompt.strip()

    if not prompt:
        continue

    print("\nPrompt:", prompt)

    inputs = tokenizer(prompt, return_tensors="pt")

    start = time.time()

    outputs = model.generate(**inputs, max_length=50)

    end = time.time()

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    latency = end - start

    print("Response:", response)
    print("Latency:", round(latency, 2), "seconds")
