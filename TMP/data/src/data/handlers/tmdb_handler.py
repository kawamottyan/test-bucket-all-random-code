from datetime import datetime
from typing import Optional, Tuple

import requests
from pydantic import ValidationError

from src.models.movie import Movie, MovieImages, MovieVideos
from src.utils.general import setup_logger

logger = setup_logger(__name__)

TMDB_TRENDING_URL = (
    "https://api.themoviedb.org/3/trending/movie/{}?language=en-US&api_key={}"
)
TMDB_MOVIE_URL = (
    "https://api.themoviedb.org/3/movie/{}?api_key={}&append_to_response=videos,images"
)


def fetch_trending_ids(tmdb_api_key: str, time_window: str = "day") -> list[int]:
    try:
        url = TMDB_TRENDING_URL.format(time_window, tmdb_api_key)
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        trending_ids = [movie["id"] for movie in data["results"]]
        return trending_ids
    except requests.RequestException as e:
        logger.error("API request failed: %s", str(e))
        return []


def fetch_movie_details(
    tmdb_api_key: str,
    tmdb_id: int,
    timestamp: datetime,
) -> Tuple[
    Optional[Movie], Optional[MovieVideos], Optional[MovieImages], Optional[dict]
]:
    try:
        url = TMDB_MOVIE_URL.format(tmdb_id, tmdb_api_key)
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        try:
            movie = Movie.model_validate(data)

        except ValidationError as e:
            failure_data = {
                "movieId": tmdb_id,
                "type": "MOVIE_VALIDATION_ERROR",
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            logger.error(
                "Movie data validation failed for TMDB ID %s: %s", tmdb_id, str(e)
            )
            return None, None, None, failure_data

        videos = None
        if data.get("videos"):
            try:
                videos_data = {
                    "movieId": data["id"],
                    "videos": data["videos"].get("results", []),
                }
                videos = MovieVideos.model_validate(videos_data)
            except ValidationError as e:
                logger.debug(
                    "Videos validation skipped for TMDB ID %s: %s", tmdb_id, str(e)
                )

        images = None
        if data.get("images"):
            try:
                images_data = {
                    "movieId": data["id"],
                    "backdrops": data["images"].get("backdrops", []),
                    "logos": data["images"].get("logos", []),
                    "posters": data["images"].get("posters", []),
                }
                images = MovieImages.model_validate(images_data)
            except ValidationError as e:
                logger.debug(
                    "Images validation skipped for TMDB ID %s: %s", tmdb_id, str(e)
                )

        return movie, videos, images, None

    except requests.RequestException as e:
        failure_data = {
            "movieId": tmdb_id,
            "type": "API_REQUEST_ERROR",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        logger.error("API request failed for TMDB ID %s: %s", tmdb_id, str(e))
        return None, None, None, failure_data
