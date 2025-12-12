import torch
import numpy as np
import faiss
import os
import gc
from PIL import Image
import base64
from io import BytesIO
import cn_clip.clip as clip
from typing import List, Tuple, Union, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt


class MultimodalRetrieverConfig():
    """
    Configuration class for Multimodal Retriever.

    Attributes:
        model_name (str): Name of the CLIP model variant (e.g., 'ViT-B-16').
        dim (int): Dimension of the CLIP embeddings (768 for ViT-B-16).
        index_path (str): Path to save or load the FAISS index.
        download_root (str): Directory for downloading CLIP models.
        batch_size (int): Batch size for processing multiple documents.
    """

    def __init__(
            self,
            model_name='ViT-B-16',
            dim=768,
            index_path='./index',
            download_root='./',
            batch_size=32
    ):
        self.model_name = model_name
        self.dim = dim
        self.index_path = index_path
        self.download_root = download_root
        self.batch_size = batch_size

    def validate(self):
        """Validate Multimodal configuration parameters."""
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("Model name must be a non-empty string.")
        if not isinstance(self.dim, int) or self.dim <= 0:
            raise ValueError("Dimension must be a positive integer.")
        if not isinstance(self.index_path, str):
            raise ValueError("Index path must be a string.")
        if not isinstance(self.download_root, str):
            raise ValueError("Download root must be a string.")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        print("Multimodal configuration is valid.")


class MultimodalRetriever():
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load_from_name(
            config.model_name,
            device=self.device,
            download_root=config.download_root
        )
        self.model.eval()
        self.dim = config.dim  # CLIP embedding dimension
        self.index = faiss.IndexFlatIP(self.dim)
        self.embeddings = []
        self.documents = []  # List to store (image_path, text) pairs
        self.num_documents = 0
        self.index_path = config.index_path
        self.batch_size = config.batch_size

    def convert_base642image(self, image_base64):
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        return image

    def merge_mm_embeddings(self, img_emb=None, text_emb=None):
        if text_emb is not None and img_emb is not None:
            return np.mean([img_emb, text_emb], axis=0)
        elif text_emb is not None:
            return text_emb
        elif img_emb is not None:
            return img_emb
        raise ValueError("Must specify one of `img_emb` or `text_emb`")

    def _embed(self, image=None, text=None) -> np.ndarray:
        if image is None and text is None:
            raise ValueError("Must specify one of image or text")

        img_emb = None
        text_emb = None

        if image is not None:
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_emb = self.model.encode_image(image)
                img_emb /= img_emb.norm(dim=-1, keepdim=True)
                img_emb = img_emb.cpu().numpy()

        if text is not None:
            text = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                text_emb = self.model.encode_text(text)
                text_emb /= text_emb.norm(dim=-1, keepdim=True)
                text_emb = text_emb.cpu().numpy()

        return self.merge_mm_embeddings(img_emb, text_emb)

    def add_image_text(self, image: Union[str, Image.Image], text: str):
        """Add a single image-text pair to the index."""
        if isinstance(image, str):
            image = self.convert_base642image(image_base64=image)

        emb = self._embed(image=image, text=text).astype('float32')
        self.index.add(emb)
        self.embeddings.append(emb)
        self.documents.append((image, text))
        self.num_documents += 1

    def build_from_pairs(self, img_text_pairs: List[Tuple[Union[str, Image.Image], str]]):
        """Build index from image-text pairs in batches."""
        if not img_text_pairs:
            raise ValueError("Image-text pairs list is empty")

        for i in tqdm(range(0, len(img_text_pairs), self.batch_size), desc="Building index"):
            batch = img_text_pairs[i:i + self.batch_size]
            for img, text in batch:
                self.add_image_text(img, text)

    def save_index(self, index_path: str = None):
        """Save the index, embeddings, and document pairs."""
        if not (self.index and self.embeddings and self.documents):
            raise ValueError("No data to save")

        if index_path is None:
            index_path = self.index_path

        os.makedirs(index_path, exist_ok=True)

        # Save embeddings and document information
        np.savez(
            os.path.join(index_path, 'multimodal.vecstore'),
            embeddings=np.array(self.embeddings),
            documents=np.array(self.documents, dtype=object)
        )

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(index_path, 'multimodal.index'))
        print(f"Index saved successfully to {index_path}")

    def load_index(self, index_path: str = None):
        """Load the index, embeddings, and document pairs."""
        if index_path is None:
            index_path = self.index_path

        # Load document data
        data = np.load(os.path.join(index_path, 'multimodal.vecstore.npz'),
                       allow_pickle=True)
        self.documents = data['documents'].tolist()
        self.embeddings = data['embeddings'].tolist()

        # Load FAISS index
        self.index = faiss.read_index(os.path.join(index_path, 'multimodal.index'))
        self.num_documents = len(self.documents)

        print(f"Index loaded successfully from {index_path}")
        del data
        gc.collect()

    def retrieve(self, query: Union[str, Image.Image], top_k: int = 5) -> List[Dict]:
        """Retrieve top_k most relevant image-text pairs."""
        if self.index is None or self.num_documents == 0:
            raise ValueError("Index is empty or not initialized")

        # Generate query embedding
        query_embedding = self._embed(
            image=query if isinstance(query, Image.Image) else None,
            text=query if isinstance(query, str) else None
        ).astype('float32')

        # Search index
        D, I = self.index.search(query_embedding, min(top_k, self.num_documents))

        # Return results with scores
        results = []
        for idx, score in zip(I[0], D[0]):
            image, text = self.documents[idx]
            results.append({
                'image': image,
                'text': text,
                'score': float(score)
            })

        return results

    def plot_results(self, query: Union[str, Image.Image], results: List[Dict], font_path: str = None):
        """
        Plot query and retrieval results with dynamic sizing and font support.

        Args:
            query: Text string or PIL Image
            results: List of retrieval results
            font_path: Path to font file for Chinese text support
        """
        # plt.close('all')  # Close any existing figures
        n_results = len(results)
        # Dynamic figure size: base width (3) for query + width for each result (3)
        figsize = (3 * (n_results + 1), 4)

        fig = plt.figure(figsize=figsize)

        # Set font for Chinese characters if provided
        if font_path:
            from matplotlib.font_manager import FontProperties
            font = FontProperties(fname=font_path)
        else:
            font = None

        # Plot query
        if isinstance(query, str):
            ax = plt.subplot(1, n_results + 1, 1)
            ax.text(0.5, 0.5, f"Query Text:\n{query}",
                    ha='center', va='center', wrap=True,
                    fontproperties=font)
            ax.axis('off')
        else:
            plt.subplot(1, n_results + 1, 1)
            plt.imshow(query)
            plt.title("Query Image", fontproperties=font)
            plt.axis('off')

        # Plot results
        for idx, result in enumerate(results, 1):
            plt.subplot(1, n_results + 1, idx + 1)
            plt.imshow(result['image'])
            plt.title(f"Score: {result['score']:.3f}\n{result['text']}",
                      pad=10, fontproperties=font)
            plt.axis('off')

        plt.tight_layout()
        # return fig
