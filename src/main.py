from typing import Optional

from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

print("Model Loaded..!")


@app.get("/")
def read_root():
    start_time = time.time()
    input_text = "Google was founded by"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]

    output = model.generate(
        input_ids,
        attention_mask=inputs["attention_mask"],
        do_sample=True,
        max_length=150,
        temperature=0.8,
        use_cache=True,
        top_p=0.9,
    )

    end_time = time.time() - start_time
    print("Total Taken => ", end_time)
    print(tokenizer.decode(output[0]))
    return {"Hello": tokenizer.decode(output[0])}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
