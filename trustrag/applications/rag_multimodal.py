import base64
from typing import List, Dict, Any
from zhipuai import ZhipuAI
from PIL import Image
from trustrag.modules.retrieval.multimodal_retriever import MultimodalRetriever,MultimodalRetrieverConfig

class MultimodalRAG:
    def __init__(
            self,
            api_key: str,
            retriever_config: MultimodalRetrieverConfig,
            model_name: str = "glm-4v-plus",
            top_k: int = 3
    ):
        self.client = ZhipuAI(api_key=api_key)
        self.retriever = MultimodalRetriever(retriever_config)
        # self.retriever.load_index()
        self.model_name = model_name
        self.top_k = top_k

    def _prepare_context(self, results: List[Dict[str, Any]]) -> str:
        context = "基于以下相似图片信息：\n"
        for idx, result in enumerate(results, 1):
            context += f"{idx}. {result['text']} (相似度: {result['score']:.2f})\n"
        return context

    def _image_to_base64(self, image: Image) -> str:
        # Convert the image to RGB mode if it's in RGBA mode
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Save the image to a BytesIO buffer in JPEG format
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")

        # Encode the image data to base64 and return it as a string
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def chat(self, query: str, include_images: bool = True) -> str:
        # 1. 检索相似内容
        results = self.retriever.retrieve(query, top_k=self.top_k)

        # 2. 准备提示信息
        context = self._prepare_context(results)
        full_prompt = f"{context}\n用户问题: {query}\n请基于用户提供的图片和上述图片信息回答问题。"

        # 3. 准备消息内容
        messages = [{"role": "user", "content": []}]

        # 4. 如果需要，添加检索到的图片
        if include_images:
            for result in results:
                img_base64 = self._image_to_base64(result['image'])
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": img_base64}
                })

        messages[0]["content"].append({"type": "text", "text": full_prompt})
        # 5. 调用API获取回答
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )

        return results, response.choices[0].message.content

    def chat_with_image(self, query: str, image_path: str) -> str:
        # 1. 读取和编码用户提供的图片
        with open(image_path, 'rb') as img_file:
            user_img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # 2. 检索相似内容
        results = self.retriever.retrieve(query, top_k=self.top_k)

        # 3. 准备提示信息
        context = self._prepare_context(results)
        full_prompt = f"{context}\n用户问题: {query}\n请基于用户提供的图片和上述相似图片信息回答问题。"

        # 4. 准备消息内容，首先添加用户的图片
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": user_img_base64}
                },
                {
                    "type": "text",
                    "text": full_prompt
                }
            ]
        }]

        # 5. 添加检索到的相似图片
        for result in results:
            img_base64 = self._image_to_base64(result['image'])
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": img_base64}
            })

        # 6. 调用API获取回答
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )

        return response.choices[0].message.content

if __name__ == '__main__':
    # 初始化配置
    car_retriever_config = MultimodalRetrieverConfig(
        model_name='ViT-B-16',
        index_path='./index_car',
        batch_size=32,
        dim=512,
        download_root="data/chinese-clip-vit-base-patch16/"
    )
    # 初始化
    car_rag = MultimodalRAG(
        api_key="xxx",
        retriever_config=car_retriever_config,
        top_k=1
    )
    query_text = "冷却系统检查"
    retrieve_results, response = car_rag.chat(query_text)
    retrieve_results, response