from io import BytesIO
from typing import List

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import ViTImageProcessor, ViTModel

from src.enums.genre import MovieGenre
from src.models.movie import Genre, Movie
from src.utils.general import setup_logger

load_dotenv()
logger = setup_logger(__name__)

TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500/"


class MovieEmbeddingHandler:
    def __init__(
        self,
        text_model_name: str,
        image_model_name: str,
        text_vector_dim: int,
        image_vector_dim: int,
    ):
        self.text_model = SentenceTransformer(text_model_name)
        self.text_vector_dim = text_vector_dim

        self.image_model = ViTModel.from_pretrained(image_model_name)
        self.image_processor = ViTImageProcessor.from_pretrained(image_model_name)
        self.image_vector_dim = image_vector_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_model.to(self.device)
        self.image_model.eval()

        self.genre_vector_dim = len(MovieGenre)

        self.total_vector_dim = (
            self.text_vector_dim + self.image_vector_dim + self.genre_vector_dim
        )

    def _get_text_embedding(self, text: str) -> np.ndarray:
        embedding = self.text_model.encode(text, convert_to_numpy=True)
        return embedding

    def _get_image_embedding(self, image_path: str) -> np.ndarray:
        try:
            if image_path.startswith("/"):
                image_path = f"{TMDB_IMAGE_BASE_URL}{image_path}"

            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            inputs = self.image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.image_model(**inputs)
                pooled_output = outputs.last_hidden_state[:, 0]
                features = pooled_output.cpu().numpy()

            return features.squeeze()
        except Exception as e:
            logger.exception("Error processing image: %s", str(e))
            return np.zeros(self.image_vector_dim)

    def _get_genre_embedding(self, genres: List[Genre]) -> np.ndarray:
        genre_vector = np.zeros(self.genre_vector_dim)
        for genre in genres:
            try:
                genre_enum = MovieGenre(genre.id)
                genre_index = list(MovieGenre).index(genre_enum)
                genre_vector[genre_index] = 1
            except ValueError:
                continue
        return genre_vector

    def _combine_embeddings(
        self,
        text_embedding: np.ndarray,
        image_embedding: np.ndarray,
        genre_embedding: np.ndarray,
    ) -> np.ndarray:
        combined = np.concatenate([text_embedding, image_embedding, genre_embedding])
        combined = combined / np.linalg.norm(combined)
        return combined

    def store_movie(self, movie: Movie) -> np.ndarray:
        overview_embedding = self._get_text_embedding(movie.overview)
        image_embedding = self._get_image_embedding(movie.posterPath)
        genre_embedding = self._get_genre_embedding(movie.genres)

        combined_embedding = self._combine_embeddings(
            overview_embedding, image_embedding, genre_embedding
        )

        return combined_embedding
