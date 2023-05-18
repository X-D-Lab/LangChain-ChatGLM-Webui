"""Wrapper around PaddleNLP embedding models."""
from typing import Any, List

from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, Extra


class PaddleNLPEmbeddings(BaseModel, Embeddings):
    """Wrapper around paddlenlp embedding models.

    To use, you should have the ``modelscope`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import PaddleNLPEmbeddings
            model = "rocketqa-zh-base-query-encoder"
            embed = PaddleNLPEmbeddings(model=model)
    """

    text_encoder: Any
    model: str ='rocketqa-zh-base-query-encoder'
    
    """Model name to use."""

    def __init__(self, **kwargs: Any):
        """Initialize the modelscope"""
        super().__init__(**kwargs)
        try:
            import paddle.nn.functional as F
            from paddlenlp import Taskflow

            self.text_encoder = Taskflow("feature_extraction", model=self.model)
            

        except ImportError as e:
            raise ValueError(
                "Could not import some python packages." "Please install it with `pip install modelscope`."
            ) from e

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a modelscope embedding model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        text_embeds = self.text_encoder(texts)
        embeddings = text_embeds["features"].numpy()

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a modelscope embedding model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        text_embeds = self.text_encoder(text)
        embedding = text_embeds["features"].numpy()[0]

        return embedding.tolist()