from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class Genre(BaseModel):
    id: int
    name: str


class ProductionCompany(BaseModel):
    id: int
    name: str
    logoPath: Optional[str] = Field(None, alias="logo_path")
    originCountry: Optional[str] = Field(None, alias="origin_country")


class ProductionCountry(BaseModel):
    iso_3166_1: str
    name: str


class SpokenLanguage(BaseModel):
    englishName: str = Field(alias="english_name")
    iso_639_1: str
    name: str


class BelongsToCollection(BaseModel):
    id: int
    name: str
    posterPath: Optional[str] = Field(None, alias="poster_path")
    backdropPath: Optional[str] = Field(None, alias="backdrop_path")


class Movie(BaseModel):
    movieId: int = Field(alias="id")
    title: str
    adult: bool
    backdropPath: Optional[str] = Field(None, alias="backdrop_path")
    belongsToCollection: Optional[BelongsToCollection] = Field(
        None, alias="belongs_to_collection"
    )
    budget: int
    genres: List[Genre]
    homepage: Optional[str] = None
    imdbId: Optional[str] = Field(None, alias="imdb_id")
    originCountry: List[str] = Field(None, alias="origin_country")
    originalLanguage: Optional[str] = Field(None, alias="original_language")
    originalTitle: Optional[str] = Field(None, alias="original_title")
    overview: str
    popularity: float
    posterPath: str = Field(alias="poster_path")
    productionCompanies: List[ProductionCompany] = Field(alias="production_companies")
    productionCountries: List[ProductionCountry] = Field(alias="production_countries")
    releaseDate: Optional[datetime] = Field(None, alias="release_date")
    revenue: int
    runtime: int
    spokenLanguages: List[SpokenLanguage] = Field(alias="spoken_languages")
    status: Optional[str] = None
    tagline: Optional[str] = None
    voteAverage: float = Field(alias="vote_average")
    voteCount: int = Field(alias="vote_count")
    video: bool


class VideoItem(BaseModel):
    videoId: str = Field(alias="id")
    iso_639_1: Optional[str]
    iso_3166_1: Optional[str]
    name: str
    key: str
    publishedAt: Optional[datetime] = Field(None, alias="published_at")
    site: str
    size: int
    type: str
    official: bool


class MovieVideos(BaseModel):
    movieId: int
    videos: List[VideoItem]


class ImageItem(BaseModel):
    aspectRatio: float = Field(alias="aspect_ratio")
    height: int
    iso_639_1: Optional[str]
    filePath: str = Field(alias="file_path")
    voteAverage: float = Field(alias="vote_average")
    voteCount: int = Field(alias="vote_count")
    width: int


class MovieImages(BaseModel):
    movieId: int
    backdrops: List[ImageItem]
    logos: List[ImageItem]
    posters: List[ImageItem]


class SafeMovieTrendings(BaseModel):
    movieId: int
    videos: List[VideoItem]
