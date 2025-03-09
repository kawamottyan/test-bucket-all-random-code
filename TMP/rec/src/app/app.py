import json
import logging
import os
import time
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pinecone import Pinecone
from pydantic import BaseModel
from upstash_redis.asyncio import Redis

from src.common.storage.s3_handler import S3Handler
from src.common.utils.general import load_config, set_random_seed, setup_logger

load_dotenv()
logger = setup_logger(__name__)

MLFLOW_BUCKET_NAME = os.getenv("MLFLOW_BUCKET_NAME", "")
AWS_REGION = os.getenv("AWS_REGION", "")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "")
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER", "")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD", "")

REDIS_URL = os.getenv("UPSTASH_REDIS_REST_URL", "")
REDIS_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_HOST = os.getenv("PINECONE_HOST", "")

REDIS_TTL = 60 * 60 * 24 * 7

config_dict: Dict[str, Any] = load_config("config.yaml")
log_level: str = config_dict.get("logging", {}).get("level", "INFO")
logger.setLevel(getattr(logging, log_level))
set_random_seed(config_dict.get("random_seed", 42))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()
s3_handler = None
redis = None
pinecone_index = None
actor_model = None

experiment_id = config_dict["mlflow"]["experiment_id"]
run_id = config_dict["mlflow"]["run_id"]
artifact_path = config_dict["mlflow"]["artifact_path"]


class Payload(BaseModel):
    model_name: str
    session_id: str
    item_size: int
    excluded_ids: List[str]


def initialize_s3_handler(config):
    if config["use_minio"]:
        return S3Handler(
            aws_region=AWS_REGION,
            bucket_name=MLFLOW_BUCKET_NAME,
            endpoint_url=MINIO_ENDPOINT,
            verify_ssl=False,
            aws_access_key_id=MINIO_ROOT_USER,
            aws_secret_access_key=MINIO_ROOT_PASSWORD,
        )
    return S3Handler(
        aws_region=AWS_REGION,
        bucket_name=MLFLOW_BUCKET_NAME,
    )


@app.on_event("startup")
async def startup_event():
    global s3_handler, redis, pinecone_index
    try:
        logger.info("Initializing S3 Handler...")
        s3_handler = initialize_s3_handler(config_dict)
        logger.info("S3 Handler initialized successfully")

        logger.info("Initializing Upstash Redis connection...")
        redis = Redis(url=REDIS_URL, token=REDIS_TOKEN)
        logger.info("Upstash Redis connection established successfully")

        logger.info("Initializing Pinecone connection...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(host=PINECONE_HOST)
        logger.info("Pinecone connection established successfully")

    except Exception as e:
        logger.error("Startup error: %s" % str(e))


@app.get("/ping")
def ping():
    return "pong"


async def get_session_data(session_id):
    if not redis:
        raise HTTPException(status_code=500, detail="Redis connection not available")

    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")

    pattern = "interaction:%s:*" % session_id
    keys = []

    cursor = 0
    while True:
        scan_result = await redis.scan(cursor, match=pattern, count=100)
        cursor = scan_result[0]
        matching_keys = scan_result[1]
        keys.extend(matching_keys)
        if cursor == 0:
            break

    if not keys:
        logger.warning("No redis keys found for session_id: %s" % session_id)
        return []

    session_data = []
    for key in keys:
        try:
            data = await redis.hgetall(key)
            if data:
                data["redis_key"] = key
                session_data.append(data)
        except Exception as e:
            logger.error("Error fetching data for key %s: %s" % (key, str(e)))

    return session_data


def get_movie_embeddings(movie_ids):
    if not pinecone_index:
        raise HTTPException(status_code=500, detail="Pinecone connection not available")

    if not movie_ids:
        return {}

    try:
        clean_ids = []
        id_mapping = {}

        for movie_id in movie_ids:
            if isinstance(movie_id, str) and ":" in movie_id:
                clean_id = movie_id.split(":")[-1]
                clean_ids.append(clean_id)
                id_mapping[clean_id] = movie_id
            else:
                clean_ids.append(str(movie_id))
                id_mapping[str(movie_id)] = movie_id

        logger.info("Looking up %d IDs in Pinecone" % len(clean_ids))

        response = pinecone_index.fetch(ids=clean_ids, namespace="movies")

        logger.debug(
            "Pinecone response keys: %s"
            % (list(response.keys() if hasattr(response, "keys") else []))
        )

        vectors_dict = {}
        if hasattr(response, "vectors"):
            for id, vector in response.vectors.items():
                vectors_dict[id] = vector.values
        else:
            for id, vector_data in response.get("vectors", {}).items():
                vectors_dict[id] = vector_data.get("values", [])

        found_ids = set(vectors_dict.keys())
        not_found_ids = set(clean_ids) - found_ids
        if not_found_ids:
            logger.warning("IDs not found in Pinecone: %s" % not_found_ids)

        embeddings = {}
        for id, values in vectors_dict.items():
            original_id = id_mapping.get(id, id)
            embeddings[original_id] = values

        logger.info(
            "Successfully retrieved %d embeddings from Pinecone" % len(embeddings)
        )
        return embeddings

    except Exception as e:
        logger.error("Error fetching movie embeddings: %s" % str(e), exc_info=True)
        return {}


async def get_similar_movies(query_vector, item_size=10, excluded_ids=None):
    if not pinecone_index:
        raise HTTPException(status_code=500, detail="Pinecone connection not available")

    if excluded_ids is None:
        excluded_ids = []

    try:
        filter_dict = None
        if excluded_ids:
            filter_dict = {"movieId": {"$nin": excluded_ids}}
            logger.info("Excluding %d IDs using metadata filter" % len(excluded_ids))

        response = pinecone_index.query(
            namespace="movies",
            vector=query_vector,
            top_k=item_size,
            include_values=True,
            filter=filter_dict,
        )

        similar_movies = []
        for match in response.matches:
            similar_movies.append(
                {
                    "movie_id": int(match.id),
                    "score": float(match.score),
                }
            )

        logger.info("Found %d similar movies" % len(similar_movies))
        return similar_movies

    except Exception as e:
        logger.error("Error fetching similar movies: %s" % str(e), exc_info=True)
        return []


async def store_recommendation_in_redis(
    session_id: str,
    action_vector: list,
    similar_movies: list,
    experiment_id: str,
    run_id: str,
):
    if not redis:
        logger.warning("Redis connection not available, skipping storage")
        return

    try:
        timestamp = int(time.time())

        key = "recommendation:%s:%d" % (session_id, timestamp)

        data = {
            "session_id": session_id,
            "timestamp": str(timestamp),
            "action_vector": json.dumps(action_vector),
            "similar_movies": json.dumps(similar_movies),
            "experiment_id": experiment_id,
            "run_id": run_id,
            "created_at": str(timestamp),
        }

        await redis.hset(key, values=data)
        await redis.expire(key, REDIS_TTL)

        logger.info(
            "Stored recommendation in Redis with key: %s, TTL: %ds" % (key, REDIS_TTL)
        )

    except Exception as e:
        logger.error("Error storing recommendation in Redis: %s" % str(e))


@app.post("/invocations")
async def invoke(payload: Payload, background_tasks: BackgroundTasks):
    global s3_handler
    try:
        if s3_handler is None:
            s3_handler = initialize_s3_handler(config_dict)

        if payload.model_name == "ddpg":
            session_data = []
            movie_embeddings = {}

            if payload.session_id:
                try:
                    session_data = await get_session_data(payload.session_id)
                    logger.info(
                        "Retrieved %d session records for session_id: %s"
                        % (len(session_data), payload.session_id)
                    )

                    movie_ids = []
                    for record in session_data:
                        item_id = record.get("item_id")
                        if item_id and item_id.startswith("movie:"):
                            movie_ids.append(item_id)

                    if movie_ids:
                        movie_embeddings = get_movie_embeddings(movie_ids)
                        logger.info(
                            "Retrieved embeddings for %d movies" % len(movie_embeddings)
                        )

                except Exception as e:
                    logger.error(
                        "Error fetching session data or embeddings: %s" % str(e)
                    )

            if len(movie_ids) < config_dict["data"]["min_items"]:
                logger.info(
                    "Number of items (%d) is less than the configured item size, returning early"
                    % len(movie_ids)
                )
                return {
                    "success": True,
                    "message": "Not enough items for recommendation",
                    "data": [],
                }

            model_path = "%s/%s/%s" % (experiment_id, run_id, artifact_path)
            logger.info("Loading model from path: %s" % model_path)
            actor_model = s3_handler.get_torch_model(model_path)
            if actor_model is None:
                raise HTTPException(
                    status_code=500, detail="Failed to load model from %s" % model_path
                )

            actor_model.eval()
            logger.info("Model loaded successfully")

            sorted_session_data = sorted(
                session_data, key=lambda x: x.get("created_at", "")
            )

            embeddings_list = []
            watch_times_list = []
            embedding_dim = config_dict["data"]["embedding_dim"]
            max_sequence_length = config_dict["data"]["max_sequence_length"]

            for record in sorted_session_data:
                item_id = record.get("item_id")
                watch_time = float(record.get("watch_time", 0))

                if item_id in movie_embeddings:
                    embedding = movie_embeddings[item_id]
                    if len(embedding) > embedding_dim:
                        embedding = embedding[:embedding_dim]
                    elif len(embedding) < embedding_dim:
                        embedding = embedding + [0] * (embedding_dim - len(embedding))

                    embeddings_list.append(embedding)
                    watch_times_list.append(watch_time)

            batch_size = 1

            sequence_length = min(len(embeddings_list), max_sequence_length)
            embeddings_list = embeddings_list[-sequence_length:]
            watch_times_list = watch_times_list[-sequence_length:]

            embeddings_tensor = torch.tensor(embeddings_list, dtype=torch.float32)
            watch_times_tensor = torch.tensor(watch_times_list, dtype=torch.float32)

            padded_embeddings = torch.zeros(
                batch_size, max_sequence_length, embedding_dim
            )
            padded_watch_times = torch.zeros(batch_size, max_sequence_length)
            mask = torch.zeros(batch_size, max_sequence_length, dtype=torch.bool)

            padded_embeddings[0, :sequence_length] = embeddings_tensor
            padded_watch_times[0, :sequence_length] = watch_times_tensor
            mask[0, :sequence_length] = 1

            state = torch.cat(
                [padded_embeddings, padded_watch_times.unsqueeze(-1)], dim=-1
            )

            state = state.to(device)
            mask = mask.to(device)
            actor_model = actor_model.to(device)

            with torch.no_grad():
                action = actor_model(state, mask)
            action_vector = action.cpu().squeeze(0).numpy().tolist()

            similar_movies = []
            error_message = ""

            if payload.item_size > 0:
                try:
                    similar_movies = await get_similar_movies(
                        query_vector=action_vector,
                        item_size=payload.item_size,
                        excluded_ids=payload.excluded_ids,
                    )
                except Exception as e:
                    error_message = "Error finding similar movies: %s" % str(e)
                    logger.error(error_message)

            if payload.session_id:
                background_tasks.add_task(
                    store_recommendation_in_redis,
                    session_id=payload.session_id,
                    action_vector=action_vector,
                    similar_movies=similar_movies,
                    experiment_id=experiment_id,
                    run_id=run_id,
                )

            return {
                "success": True,
                "message": error_message if error_message else "Success",
                "data": similar_movies,
            }
    except Exception as e:
        error_message = "Inference error: %s" % str(e)
        logger.error(error_message, exc_info=True)
        return {
            "success": False,
            "message": error_message,
            "data": [],
            "debug_info": {"session_id": getattr(payload, "session_id", None)},
        }
