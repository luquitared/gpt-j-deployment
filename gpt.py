from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class Output(BaseModel):
    result: str


model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

@app.post("/")
async def root(output: Output):
    print(output)
    prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
         "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
         "researchers was the fact that the unicorns spoke perfect English."

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    print(gen_text)
    return {"message": gen_text}