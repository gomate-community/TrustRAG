from trustrag.modules.generator.chat import ZhipuChat

if __name__ == '__main__':
    # 初始化 ZhipuChat
    zhipu_chat = ZhipuChat(key="your_zhipuai_api_key")

    # 定义系统提示和用户消息
    system_message = "你是一个聪明且富有创造力的小说作家"
    user_message = "请你作为童话故事大王，写一篇短篇童话故事，故事的主题是要永远保持一颗善良的心，要能够激发儿童的学习兴趣和想象力，同时也能够帮助儿童更好地理解和接受故事中所蕴含的道理和价值观。"

    # 调用 chat 方法
    response, total_tokens = zhipu_chat.chat(
        system=system_message,
        history=[{"role": "user", "content": user_message}],
        gen_conf={"top_p": 0.7, "temperature": 0.9}
    )

    # 打印响应
    print(response)
    print(f"Total tokens used: {total_tokens}")