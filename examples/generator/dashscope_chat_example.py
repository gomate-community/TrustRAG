from trustrag.modules.generator.chat import DashScopeChat

if __name__ == '__main__':
    # 初始化 DashScopeChat
    dashscope_chat = DashScopeChat(key="DASHSCOPE_API_KEY")

    # 定义系统提示和用户消息
    system_message = "You are a helpful assistant."
    user_message = "你是谁？"

    # 调用 chat 方法
    response, total_tokens = dashscope_chat.chat(
        system=system_message,
        history=[{"role": "user", "content": user_message}],
        gen_conf={}  # 可以根据需要添加生成配置，如 top_p 和 temperature
    )

    # 打印响应
    print(response)
    print(f"Total tokens used: {total_tokens}")