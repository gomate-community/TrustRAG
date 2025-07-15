#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: llm.py
@time: 2024/05/16
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import os
from typing import Dict, List, Any

import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from trustrag.modules.prompt.templates import SYSTEM_PROMPT, CHAT_PROMPT_TEMPLATES



class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass


class OpenAIChat(BaseModel):
    def __init__(self, path: str = '', model: str = "gpt-3.5-turbo-1106") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")
        client.base_url = os.getenv("OPENAI_BASE_URL")
        history.append({'role': 'user',
                        'content': CHAT_PROMPT_TEMPLATES['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)})
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message.content


class InternLMChat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history: List = [], content: str = '') -> str:
        prompt = CHAT_PROMPT_TEMPLATES['InternLM_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        response, history = self.model.chat(self.tokenizer, prompt, history)
        return response

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16,
                                                          trust_remote_code=True).cuda()


class GLM3Chat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history=None, content: str = '', llm_only: bool = False) -> tuple[Any, Any]:
        if history is None:
            history = []
        if llm_only:
            prompt = prompt
        else:
            prompt = CHAT_PROMPT_TEMPLATES['GLM_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        response, history = self.model.chat(self.tokenizer, prompt, history, max_length=32000, num_beams=1,
                                            do_sample=True, top_p=0.8, temperature=0.2)
        return response, history

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16,
                                                          trust_remote_code=True).cuda()


class GLM4Chat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history=None, content: str = '', llm_only: bool = False) -> tuple[Any, Any]:
        if llm_only:
            prompt = prompt
        else:
            prompt = CHAT_PROMPT_TEMPLATES['GLM_PROMPT_TEMPALTE'].format(system_prompt=SYSTEM_PROMPT, question=prompt,
                                                                    context=content)
        prompt = prompt.encode("utf-8", 'ignore').decode('utf-8', 'ignore')
        print(prompt)

        inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                    add_generation_prompt=True,
                                                    tokenize=True,
                                                    return_tensors="pt",
                                                    return_dict=True
                                                    )

        inputs = inputs.to('cuda')
        gen_kwargs = {"max_length": 5120, "do_sample": False, "top_k": 1}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response, history = output, []
            return response, history

    def load_model(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).cuda().eval()


class QwenChat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()
        self.device = 'cuda'

    def chat(self, prompt: str, history: List = [], content: str = '', llm_only: bool = False) -> tuple[Any, Any]:
        if llm_only:
            prompt = prompt
        else:
            prompt = CHAT_PROMPT_TEMPLATES['DF_QWEN_PROMPT_TEMPLATE2'].format(question=prompt, context=content)
        # print(prompt)
        messages = [
            {"role": "system", "content": "你是一个专门用于回答中国电信运营商相关问题的AI助手。"},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(text)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            do_sample=False,
            top_k=10
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response, history

    def load_model(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16,
        #                                                   trust_remote_code=True).cuda()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("load model success")


# !/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: llm.py
@time: 2024/05/16
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import os
from typing import Dict, List, Any

import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from trustrag.modules.prompt.templates import SYSTEM_PROMPT, CHAT_PROMPT_TEMPLATES


class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass


class OpenAIChat(BaseModel):
    def __init__(self, path: str = '', model: str = "gpt-3.5-turbo-1106") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        client = OpenAI()
        client.api_key = os.getenv("OPENAI_API_KEY")
        client.base_url = os.getenv("OPENAI_BASE_URL")
        history.append({'role': 'user',
                        'content': CHAT_PROMPT_TEMPLATES['RAG_PROMPT_TEMPALTE'].format(question=prompt,
                                                                                       context=content)})
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message.content


class InternLMChat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history: List = [], content: str = '') -> str:
        prompt = CHAT_PROMPT_TEMPLATES['InternLM_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        response, history = self.model.chat(self.tokenizer, prompt, history)
        return response

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16,
                                                          trust_remote_code=True).cuda()


class GLM3Chat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history=None, content: str = '', llm_only: bool = False) -> tuple[Any, Any]:
        if history is None:
            history = []
        if llm_only:
            prompt = prompt
        else:
            prompt = CHAT_PROMPT_TEMPLATES['GLM_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        response, history = self.model.chat(self.tokenizer, prompt, history, max_length=32000, num_beams=1,
                                            do_sample=True, top_p=0.8, temperature=0.2)
        return response, history

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, torch_dtype=torch.float16,
                                                          trust_remote_code=True).cuda()


class GLM4Chat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()

    def chat(self, prompt: str, history=None, content: str = '', llm_only: bool = False) -> tuple[Any, Any]:
        if llm_only:
            prompt = prompt
        else:
            prompt = CHAT_PROMPT_TEMPLATES['GLM_PROMPT_TEMPALTE'].format(system_prompt=SYSTEM_PROMPT, question=prompt,
                                                                         context=content)
        prompt = prompt.encode("utf-8", 'ignore').decode('utf-8', 'ignore')
        print(prompt)

        inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                    add_generation_prompt=True,
                                                    tokenize=True,
                                                    return_tensors="pt",
                                                    return_dict=True
                                                    )

        inputs = inputs.to('cuda')
        gen_kwargs = {"max_length": 5120, "do_sample": False, "top_k": 1}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response, history = output, []
            return response, history

    def load_model(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).cuda().eval()


class Qwen3Chat(BaseModel):
    def __init__(self, path: str = '') -> None:
        super().__init__(path)
        self.load_model()
        self.device = 'cuda'

    def chat(self, prompt: str, history: List = [], content: str = '', llm_only: bool = False,
             enable_thinking: bool = True) -> tuple[Any, Any]:
        if llm_only:
            prompt = prompt
        else:
            # 使用适当的prompt模板，可以根据需要调整
            prompt = CHAT_PROMPT_TEMPLATES.get('DF_QWEN_PROMPT_TEMPLATE2', '{question}\n\n上下文：{context}').format(
                question=prompt, context=content)

        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking  # 支持thinking模式
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 生成文本，支持更大的token数量
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768,  # 支持更大的生成长度
            do_sample=False,
            top_k=10
        )

        # 提取生成的部分
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        return response, history

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype="auto",  # 使用auto自动选择最佳数据类型
            device_map="auto",  # 自动设备映射
            trust_remote_code=True
        )
        print("Qwen3 model loaded successfully")


