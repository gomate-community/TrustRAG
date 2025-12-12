from trustrag.modules.generator.chat import DeepSeekChat

if __name__ == '__main__':

    api_key = "sk-04031f18c05a4dd5a561d33d984ca40f"  # 替换为你的 DeepSeek API Key
    deepseek_chat = DeepSeekChat(key=api_key)

    system_prompt = "You are a helpful assistant."
    history = [
        {"role": "user", "content": "Hello"}
    ]
    gen_conf = {
        "temperature": 0.7,
        "max_tokens": 100
    }

    # 调用 chat 方法进行对话

    response, total_tokens = deepseek_chat.chat(system=system_prompt, history=history, gen_conf=gen_conf)
    print("Response:", response)
    print("Total Tokens:", total_tokens)

    # 调用 chat_streamly 方法进行流式对话

    for response in deepseek_chat.chat_streamly(system=system_prompt, history=history, gen_conf=gen_conf):
        if isinstance(response, str):
            print("Stream Response:", response)
        else:
            print("Total Tokens:", response)