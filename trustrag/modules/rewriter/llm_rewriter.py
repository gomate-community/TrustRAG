from trustrag.modules.generator.chat import OpenAIChat

DEFAULT_SYSTEM_PROMPT='''你是一名搜索优化专家，擅长改写用户查询，使其更适合搜索引擎处理。'''
DEFAULT_REWRITE_PROMPT='''
请基于以下用户输入的问题，扩展出多个相关的搜索词组，确保：  
1. 只扩展关键词，不改写为完整的问题句。  
2. 生成 3-5 组相关的搜索关键词，覆盖不同的搜索角度。  
3. 结合同义词、近义词、行业术语、相关概念，确保搜索范围更广。  
4. 结果用中文分号（`;`）分隔，不包含多余的解释或符号。  
5.关键词之间的词语可以用空格隔开，显示有层次感
**用户查询：**  
"{query}"  

**扩展后的关键词组：**
'''
class LLMRewriter():
    def __init__(self,api_key=None,model_name="",base_url=None,system_prompt=DEFAULT_SYSTEM_PROMPT,rewrite_prompt=DEFAULT_REWRITE_PROMPT):
        self.api_key = api_key
        self.chat = OpenAIChat(key=api_key,model_name=model_name,base_url=base_url)
        self.system_prompt = system_prompt
        self.rewrite_prompt = rewrite_prompt
    def rewrite(self, query):
        prompt=self.rewrite_prompt.format(query=query)
        history = [
            {"role": "user", "content": prompt}
        ]
        gen_conf = {
            "temperature": 0.1,
        }
        # 调用 chat 方法进行对话
        response, total_tokens = self.chat.chat(system= self.system_prompt, history=history, gen_conf=gen_conf)
        return response


if __name__ == '__main__':
    llm_rewriter = LLMRewriter(api_key='sk-gDbFoQAYz9pwqBsH0aPA1H8DN9s0B9F3vWNjjPcijRBFjk7f')
    # response=llm_rewriter.rewrite(query="如何提高英语口语？")
    response=llm_rewriter.rewrite(query="请总结伊朗总统罹难事件")
    print(response)