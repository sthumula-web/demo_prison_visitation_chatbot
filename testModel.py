from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 



model_path = "/Users/sisirathumula/programs/CORRECT_demo/phi-4-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map="mps"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

# Apply chat template to get the formatted text
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Tokenize separately
inputs = tokenizer(text, return_tensors="pt").to("mps")
print ("debug1")
outputs = model.generate(**inputs, max_new_tokens=20)
print ("debug2")
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print(response)






