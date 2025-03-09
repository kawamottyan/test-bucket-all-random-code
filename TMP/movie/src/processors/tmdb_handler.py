from typing import Tuple, Optional
import requests
import logging
from pydantic import ValidationError
from src.models.movie import Movie, MovieVideos, MovieImages

logger = logging.getLogger(__name__)

TMDB_TRENDING_URL = "https://api.themoviedb.org/3/trending/movie/{}?api_key={}"
TMDB_MOVIE_URL = (
    "https://api.themoviedb.org/3/movie/{}?api_key={}&append_to_response=videos,images"
)


def fetch_trending_movie_ids(
    tmdb_api_key: str, time_window: str = "day"
) -> Tuple[Optional[Movie], Optional[MovieVideos], Optional[MovieImages]]:
    url = TMDB_TRENDING_URL.format(time_window, tmdb_api_key)
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        trending_ids = [movie["id"] for movie in data["results"]]
        return trending_ids
    else:
        print(f"Failed to fetch trending movies: {response.status_code}")
        return []


def fetch_movie_details(
    tmdb_api_key: str,
    tmdb_id: int,
    timestamp: str,
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
                "movieId": str(tmdb_id),
                "type": "MOVIE_VALIDATION_ERROR",
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            logger.error(f"Movie data validation failed for TMDB ID {tmdb_id}: {e}")
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
                logger.debug(f"Videos validation skipped for TMDB ID {tmdb_id}: {e}")

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
                logger.debug(f"Images validation skipped for TMDB ID {tmdb_id}: {e}")

        return movie, videos, images, None

    except requests.RequestException as e:
        failure_data = {
            "movieId": str(tmdb_id),
            "type": "API_REQUEST_ERROR",
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        logger.error(f"API request failed for TMDB ID {tmdb_id}: {e}")
        return None, None, None, failure_data
