from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    #device_map="auto"
)

model = model.to("cuda" if torch.cuda.is_available() else "cpu")

goal = "Make tea"
history = ["Pick up kettle", "Go to sink", "Fill kettle"]
prompt = f"""[INST] <<SYS>>
You are an intelligent planner that helps users achieve goals by predicting the next logical step.
<</SYS>>

Goal: {goal}
Steps so far: {', '.join(history)}
What is the next step?
[/INST]
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated step:", response.split('[/INST]')[-1].strip())
