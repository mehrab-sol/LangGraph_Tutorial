from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

# First API call
response = client.chat.completions.create(
    model="minimax/minimax-m2.5:free",
    messages=[
        {
            "role": "user",
            "content": "How many r's are in the word 'strawberry'?"
        }
    ],
    extra_body={"reasoning": {"enabled": True}}
)

print("First response:")
print(response.choices[0].message.content)

first_message = response.choices[0].message

messages = [
    {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
    {
        "role": "assistant",
        "content": first_message.content,
        "reasoning_details": first_message.reasoning_details
    },
    {"role": "user", "content": "Are you sure? Think carefully."}
]

# Second API call
response2 = client.chat.completions.create(
    model="minimax/minimax-m2.5:free",
    messages=messages,
    extra_body={"reasoning": {"enabled": True}}
)

print("\nSecond response:")
print(response2.choices[0].message.content)