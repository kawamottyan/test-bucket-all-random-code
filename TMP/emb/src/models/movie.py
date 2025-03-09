from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Genre(BaseModel):
    id: int
    name: str


class ProductionCompany(BaseModel):
    id: int
    name: str
    logoPath: Optional[str]
    originCountry: Optional[str]


class ProductionCountry(BaseModel):
    iso_3166_1: str
    name: str


class SpokenLanguage(BaseModel):
    englishName: str
    iso_639_1: str
    name: str


class BelongsToCollection(BaseModel):
    id: int
    name: str
    posterPath: Optional[str]
    backdropPath: Optional[str]


class Movie(BaseModel):
    movieId: int
    title: str
    adult: bool
    backdropPath: Optional[str]
    belongsToCollection: Optional[BelongsToCollection]
    budget: int
    genres: List[Genre]
    homepage: Optional[str] = None
    imdbId: Optional[str]
    originCountry: List[str]
    originalLanguage: Optional[str]
    originalTitle: Optional[str]
    overview: str
    popularity: float
    posterPath: str
    productionCompanies: List[ProductionCompany]
    productionCountries: List[ProductionCountry]
    releaseDate: Optional[datetime]
    revenue: int
    runtime: int
    spokenLanguages: List[SpokenLanguage]
    status: Optional[str] = None
    tagline: Optional[str] = None
    voteAverage: float
    voteCount: int
    video: bool


class VideoItem(BaseModel):
    videoId: str
    iso_639_1: Optional[str]
    iso_3166_1: Optional[str]
    name: str
    key: str
    publishedAt: Optional[datetime]
    site: str
    size: int
    type: str
    official: bool


class MovieVideos(BaseModel):
    movieId: int
    videos: List[VideoItem]


class ImageItem(BaseModel):
    aspectRatio: float
    height: int
    iso_639_1: Optional[str]
    filePath: str
    voteAverage: float
    voteCount: int
    width: int


class MovieImages(BaseModel):
    movieId: int
    backdrops: List[ImageItem]
    logos: List[ImageItem]
    posters: List[ImageItem]


class MovieTrendings(BaseModel):
    movieId: int
    videos: List[VideoItem]
