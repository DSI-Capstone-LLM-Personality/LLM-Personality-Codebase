import openai

openai.api_key = 'api_key'

def gpt3(prompt):
    response = openai.Completion.create(
      model="text-curie-001",
      prompt=prompt,
      temperature=0.7,
      max_tokens=64,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    res = response['choices'][0]['text'].strip()
    return res
print(gpt3('what is 10+10'))

