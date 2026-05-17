from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

model.eval()
prompt = """
Faculty of Mathematics and Informatics organizes courses in programming,
artificial intelligence, data analysis and software engineering.
Students participate in projects and competitions.
"""
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(
    input_ids,
    max_length=180,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    do_sample=True
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("GENERATED TEXT\n")
print(generated_text)