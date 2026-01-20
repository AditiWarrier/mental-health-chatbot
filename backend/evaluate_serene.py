from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

MODEL_DIR = "./serene_model"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

model.eval()

print("Serene evaluator. Type messages (CTRL+C to quit).")

while True:
    try:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        prompt = f"User: {user_input}\nSerene:"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=120,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if "Serene:" in text:
            serene_reply = text.split("Serene:", 1)[1].strip()
        else:
            serene_reply = text.strip()

        print(f"Serene: {serene_reply}\n")

    except KeyboardInterrupt:
        print("\nExiting.")
        break
