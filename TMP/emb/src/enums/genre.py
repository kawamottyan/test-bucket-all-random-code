from enum import IntEnum
from typing import Dict


class MovieGenre(IntEnum):
    ACTION = 28
    ADVENTURE = 12
    ANIMATION = 16
    COMEDY = 35
    CRIME = 80
    DOCUMENTARY = 99
    DRAMA = 18
    FAMILY = 10751
    FANTASY = 14
    HISTORY = 36
    HORROR = 27
    MUSIC = 10402
    MYSTERY = 9648
    ROMANCE = 10749
    SCIENCE_FICTION = 878
    TV_MOVIE = 10770
    THRILLER = 53
    WAR = 10752
    WESTERN = 37

    @classmethod
    def get_name(cls, genre_id: int) -> str:
        genre_names: Dict[MovieGenre, str] = {
            cls.ACTION: "Action",
            cls.ADVENTURE: "Adventure",
            cls.ANIMATION: "Animation",
            cls.COMEDY: "Comedy",
            cls.CRIME: "Crime",
            cls.DOCUMENTARY: "Documentary",
            cls.DRAMA: "Drama",
            cls.FAMILY: "Family",
            cls.FANTASY: "Fantasy",
            cls.HISTORY: "History",
            cls.HORROR: "Horror",
            cls.MUSIC: "Music",
            cls.MYSTERY: "Mystery",
            cls.ROMANCE: "Romance",
            cls.SCIENCE_FICTION: "Science Fiction",
            cls.TV_MOVIE: "TV Movie",
            cls.THRILLER: "Thriller",
            cls.WAR: "War",
            cls.WESTERN: "Western",
        }
        return genre_names.get(MovieGenre(genre_id), "Unknown")

    @classmethod
    def get_all_genres(cls) -> Dict[int, str]:
        return {genre.value: cls.get_name(genre.value) for genre in cls}
