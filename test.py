import transformers
import torch
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, AutoTokenizer
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7b")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7b")

with torch.no_grad():
    # Text completion
    prompt = "Given a statement of you: you make friend easily. What do you think?\nVery Accurate\nModerately Accurate\nNeither Accurate Nor Inaccurate\nModerately Inaccurate\nVery Inaccurate\nAnswer: "

    # print(eg_q)
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids
    response = model.generate(input_ids,top_p=0.95, temperature=0.1,
                                   max_new_tokens=70)
    output = tokenizer.decode(response[0])
print(output)
print(len(response[0]))