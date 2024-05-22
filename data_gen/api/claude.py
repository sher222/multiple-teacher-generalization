import anthropic

client = anthropic.Anthropic(api_key = "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")
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)
print(message.content)