from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from typing import List
from sklearn.preprocessing import PolynomialFeatures
import torch
import os

# 环境变量传入
sk_key = os.environ.get("sk-key", "sk-aaabbbcccdddeeefffggghhhiiijjjkkk")

# 创建FastAPI实例
app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建HTTPBearer实例
security = HTTPBearer()


# 预加载模型
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"本次加载模型的设备为：{'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU.'}"
    )
    return SentenceTransformer("./models/bge-large-zh-v1.5", device=device)


model = load_model()


# 请求和响应模型
class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict


# 功能函数
def process_embedding(embedding, target_length):
    """扩展或截断嵌入向量以匹配目标长度。"""
    poly = PolynomialFeatures(degree=2)
    expanded_embedding = poly.fit_transform(embedding.reshape(1, -1)).flatten()

    if len(expanded_embedding) > target_length:
        return expanded_embedding[:target_length]
    elif len(expanded_embedding) < target_length:
        return np.pad(expanded_embedding, (0, target_length - len(expanded_embedding)))
    return expanded_embedding


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(
    request: EmbeddingRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials.credentials != sk_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization code",
        )

    # 计算嵌入向量
    embeddings = [
        model.encode(text, normalize_embeddings=True) for text in request.input
    ]
    embeddings = [process_embedding(embedding, 1536) for embedding in embeddings]

    # Min-Max normalization
    embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]

    # 转换为列表
    embeddings = [embedding.tolist() for embedding in embeddings]
    prompt_tokens = sum(len(text.split()) for text in request.input)
    total_tokens = sum(len(text) for text in request.input)  # 示例：计算总tokens数量

    response = {
        "data": [
            {"embedding": embedding, "index": index, "object": "embedding"}
            for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "object": "list",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        },
    }

    return response


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=6008, workers=1)