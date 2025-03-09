from typing import Any, Dict, List, Mapping, Tuple, Union

import polars as pl

from src.models.log import (
    InteractionType,
)

SchemaType = Mapping[str, Union[pl.DataType, type[pl.DataType]]]


def validate_schema(df: pl.DataFrame, schema: SchemaType) -> Tuple[bool, List[str]]:
    errors = []
    for col, expected_dtype in schema.items():
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
            continue

        series = df[col]
        if col in ("synced_at", "embedded_at"):
            non_null_series = series.drop_nulls()
            if non_null_series.shape[0] > 0:
                if non_null_series.dtype != expected_dtype:
                    errors.append(
                        f"Column {col} has wrong type: expected {expected_dtype}, got {non_null_series.dtype}"
                    )
        else:
            if series.dtype != expected_dtype:
                errors.append(
                    f"Column {col} has wrong type: expected {expected_dtype}, got {series.dtype}"
                )
    return (len(errors) == 0), errors


movies_schema: Dict[str, Union[pl.DataType, type[pl.DataType]]] = {
    "movieId": pl.Int64,
    "title": pl.Utf8,
    "adult": pl.Boolean,
    "backdropPath": pl.Utf8,
    "budget": pl.Int64,
    "genres": pl.List(pl.Struct({"id": pl.Int64, "name": pl.Utf8})),
    "homepage": pl.Utf8,
    "imdbId": pl.Utf8,
    "originCountry": pl.List(pl.Utf8),
    "originalLanguage": pl.Utf8,
    "originalTitle": pl.Utf8,
    "overview": pl.Utf8,
    "popularity": pl.Float64,
    "posterPath": pl.Utf8,
    "productionCompanies": pl.List(
        pl.Struct(
            {
                "id": pl.Int64,
                "name": pl.Utf8,
                "logoPath": pl.Utf8,
                "originCountry": pl.Utf8,
            }
        )
    ),
    "productionCountries": pl.List(pl.Struct({"iso_3166_1": pl.Utf8, "name": pl.Utf8})),
    "releaseDate": pl.Datetime,
    "revenue": pl.Int64,
    "runtime": pl.Int64,
    "spokenLanguages": pl.List(
        pl.Struct({"englishName": pl.Utf8, "iso_639_1": pl.Utf8, "name": pl.Utf8})
    ),
    "status": pl.Utf8,
    "tagline": pl.Utf8,
    "voteAverage": pl.Float64,
    "voteCount": pl.Int64,
    "video": pl.Boolean,
    "belongsToCollection": pl.Struct(
        {
            "id": pl.Int64,
            "name": pl.Utf8,
            "posterPath": pl.Utf8,
            "backdropPath": pl.Utf8,
        }
    ),
    "created_at": pl.Datetime,
    "updated_at": pl.Datetime,
    "synced_at": pl.Datetime,
    "embedded_at": pl.Datetime,
}

videos_schema: Dict[str, Union[pl.DataType, type[pl.DataType]]] = {
    "videoId": pl.Utf8,
    "iso_639_1": pl.Utf8,
    "iso_3166_1": pl.Utf8,
    "name": pl.Utf8,
    "key": pl.Utf8,
    "publishedAt": pl.Datetime,
    "site": pl.Utf8,
    "size": pl.Int64,
    "type": pl.Utf8,
    "official": pl.Boolean,
    "movieId": pl.Int64,
    "created_at": pl.Datetime,
    "updated_at": pl.Datetime,
    "synced_at": pl.Datetime,
    "embedded_at": pl.Datetime,
}

images_schema: Dict[str, Union[pl.DataType, type[pl.DataType]]] = {
    "aspectRatio": pl.Float64,
    "height": pl.Int64,
    "iso_639_1": pl.Utf8,
    "filePath": pl.Utf8,
    "voteAverage": pl.Float64,
    "voteCount": pl.Int64,
    "width": pl.Int64,
    "movieId": pl.Int64,
    "imageType": pl.Utf8,
    "created_at": pl.Datetime,
    "updated_at": pl.Datetime,
    "synced_at": pl.Datetime,
    "embedded_at": pl.Datetime,
}

failures_schema: Dict[str, Union[pl.DataType, type[pl.DataType]]] = {
    "movieId": pl.Utf8,
    "type": pl.Utf8,
    "created_at": pl.Datetime,
    "updated_at": pl.Datetime,
}

trendings_schema: Dict[str, Union[pl.DataType, type[pl.DataType]]] = {
    "movieIds": pl.List(pl.Int64),
    "created_at": pl.Datetime,
    "updated_at": pl.Datetime,
    "synced_at": pl.Datetime,
}

log_schemas = {
    InteractionType.POSTER_VIEWED: {
        "uuid": pl.Utf8,
        "interaction_type": pl.Utf8,
        "item_id": pl.Utf8,
        "watch_time": pl.Utf8,
        "query": pl.Utf8,
        "index": pl.Utf8,
        "created_at": pl.Datetime,
        "local_timestamp": pl.Utf8,
        "session_id": pl.Utf8,
        "migrated_at": pl.Datetime,
    },
    InteractionType.DETAIL_VIEWED: {
        "uuid": pl.Utf8,
        "interaction_type": pl.Utf8,
        "item_id": pl.Utf8,
        "query": pl.Utf8,
        "index": pl.Utf8,
        "created_at": pl.Datetime,
        "local_timestamp": pl.Utf8,
        "session_id": pl.Utf8,
        "migrated_at": pl.Datetime,
    },
    InteractionType.PLAY_STARTED: {
        "uuid": pl.Utf8,
        "interaction_type": pl.Utf8,
        "item_id": pl.Utf8,
        "watch_time": pl.Utf8,
        "query": pl.Utf8,
        "index": pl.Utf8,
        "created_at": pl.Datetime,
        "local_timestamp": pl.Utf8,
        "session_id": pl.Utf8,
        "migrated_at": pl.Datetime,
    },
}
