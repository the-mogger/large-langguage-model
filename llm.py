import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import random
from flask import Flask, request, jsonify

#  Load the actual GPT-2 model and tokenizer
model_name = "gpt2"  # Can change to 'gpt2-medium' if you got more VRAM
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

#  Load dataset
dataset = load_dataset("text", data_files={"train": "verdict.txt"})

# Custom Dataset class
class VerdictDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.inputs.items()}

#  Tokenize dataset
texts = [example["text"] for example in dataset["train"]]
dataset = VerdictDataset(texts, tokenizer)
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Batch size can be changed based on GPU

#  Training Setup
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Can increase for better fine-tuning
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    fp16=torch.cuda.is_available()  # Use mixed precision if on GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

#  Start Fine-Tuning
trainer.train()

# Save model & tokenizer after training
model.save_pretrained("gpt2_finetuned_verdict")
tokenizer.save_pretrained("gpt2_finetuned_verdict")

#  Load fine-tuned model
model = GPT2LMHeadModel.from_pretrained("gpt2_finetuned_verdict")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#  Function to Generate Text
def generate_text(prompt, max_length=50, temperature=1.0, top_k=50, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids, 
        max_length=max_length, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p, 
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

#  Test generation
print(generate_text("The verdict is"))

#  Flask API for Text Generation
app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt", "The verdict is")
    temperature = float(data.get("temperature", 1.0))
    top_k = int(data.get("top_k", 50))
    top_p = float(data.get("top_p", 0.9))

    result = generate_text(prompt, temperature=temperature, top_k=top_k, top_p=top_p)
    return jsonify({"generated_text": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
