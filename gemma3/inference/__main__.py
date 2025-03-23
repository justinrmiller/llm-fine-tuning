from unsloth.chat_templates import get_chat_template

from unsloth import FastModel

model_name = ""

model, tokenizer = FastModel.from_pretrained(
    model_name = model_name, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    load_in_4bit = True,
)


tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)
messages = [{
    "role": "user",
    "content": [{
        "type" : "text",
        "text" : "Continue the sequence: 1, 1, 2, 3, 5, 8,",
    }]
}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
)
outputs = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
)

print(tokenizer.batch_decode(outputs))
