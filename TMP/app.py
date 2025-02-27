import os
from typing import List
import logging
from dotenv import load_dotenv
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.models.data_models import Movie
from src.utils.helpers import load_config, set_random_seed
from src.processors.s3_handler import S3Handler
from src.processors.movie_processing import filter_input_movies, get_latest_n_movies
from src.features.embedding import create_stacked_embedding
from src.processors.movie_recommendation import get_top_n_recommended_items

logger = logging.getLogger(__name__)
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")
EMBEDDING_KEY = os.getenv("EMBEDDING_KEY")


class Payload(BaseModel):
    input: List[Movie] = Field(...)
    ml_model_name: str = Field(...)
    ml_model_type_name: str = Field(...)
    recommendation_size: int = Field(10, gt=0)
    excluded_tmdbIds: List[int] = Field(default=[])


app = FastAPI()


@app.get("/ping")
def ping():
    return "pong"


@app.post("/invocations")
async def invoke(payload: Payload):
    try:
        config = load_config("config.yaml")
        logging.basicConfig(level=config.get("logging_level_name", "INFO").upper())
        set_random_seed(config.get("random_seed", 42))

        input_data = payload.input
        model_name = payload.ml_model_name
        model_type_name = payload.ml_model_type_name
        item_size = payload.recommendation_size
        excluded_Ids = payload.excluded_tmdbIds

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s3_handler = S3Handler(AWS_REGION)

        embedding_df = s3_handler.get_s3_data(
            BUCKET_NAME, EMBEDDING_KEY, file_format="parquet"
        )

        norm_stacked_embeddings = create_stacked_embedding(
            embedding_df, config["embedding_size"]
        ).to(device)

        index_mapping_dict = dict(
            zip(embedding_df["tmdbId"].to_list(), embedding_df["itemIndex"].to_list())
        )
        id_mapping_dict = dict(
            zip(embedding_df["itemIndex"].to_list(), embedding_df["tmdbId"].to_list())
        )

        movie_list = filter_input_movies(input_data, index_mapping_dict)
        indices, watchTimes = get_latest_n_movies(
            movie_list, index_mapping_dict, n=config["frame_size"]
        )

        capped_watchTimes = [min(watchTime, 30) for watchTime in watchTimes]
        if model_type_name == "ddpg":
            actor_model = s3_handler.get_s3_data(
                BUCKET_NAME,
                f"model_storage/{model_type_name}/{model_name}.pt",
                file_format="pt",
            )
            actor_model = actor_model.to(device)
            tensor_item = torch.tensor(indices).unsqueeze(0).to(device)
            tensor_watchTimes = (
                torch.tensor(capped_watchTimes).float().unsqueeze(0).to(device)
            )
            tensor_embedding = (
                norm_stacked_embeddings[tensor_item.long()].float().to(device)
            )
            flattened_tensor_embedding = tensor_embedding.view(1, -1).to(device)

            state = torch.cat([flattened_tensor_embedding, tensor_watchTimes], 1)

            with torch.no_grad():
                predicted_action = actor_model(state)

            norm_predicted_action = F.normalize(predicted_action, p=2, dim=1)
            cosine_sim = torch.matmul(
                norm_predicted_action, norm_stacked_embeddings.t()
            )
            _, top_k_indices = torch.topk(cosine_sim, cosine_sim.size(1), dim=1)

            recommended_items = get_top_n_recommended_items(
                top_k_indices,
                cosine_sim,
                indices,
                id_mapping_dict,
                excluded_Ids,
                item_size,
            )

            return {
                "status": "success",
                "output": recommended_items,
                "model_name": model_name,
            }
        else:
            raise ValueError(f"Unsupported model_type_name: {model_type_name}")

    except Exception as e:
        logger.error(f"Error during request processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
