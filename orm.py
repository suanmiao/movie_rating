from peewee import *
import time
import datetime
from playhouse.fields import ManyToManyField
import util
import os

from playhouse.db_url import connect

USER_NAME = 'suan'
PWD = os.getenv('PWD_CLASSIC', '1354')
DB_NAME = 'crawler'
PORT = '3306'
url = 'mysql://' + USER_NAME + ':' + PWD + '@104.131.78.71:' + PORT + '/' + DB_NAME
print url
db = connect(url)


class CharacterModel(Model):
    subject_id = CharField(unique=True, max_length=400)
    name = CharField(max_length=800)

    class Meta:
        global db
        database = db  # This model uses the "people.db" database.


class ActorModel(CharacterModel):
    class Meta:
        global db
        database = db  # This model uses the "people.db" database.


class DirectorModel(CharacterModel):
    class Meta:
        global db
        database = db  # This model uses the "people.db" database.


class WriterModel(CharacterModel):
    class Meta:
        global db
        database = db  # This model uses the "people.db" database.


class GenreModel(Model):
    name = CharField(unique=True, max_length=800)

    class Meta:
        global db
        database = db  # This model uses the "people.db" database.


class CountryModel(Model):
    name = CharField(unique=True, max_length=800)

    class Meta:
        global db
        database = db  # This model uses the "people.db" database.


class MovieModel(Model):
    subject_id = CharField(unique=True, max_length=400)
    title = CharField(max_length=800)
    genres = ManyToManyField(GenreModel, related_name='genres')
    countries = ManyToManyField(CountryModel, related_name='countries')
    actors = ManyToManyField(ActorModel, related_name='actors')
    writers = ManyToManyField(WriterModel, related_name='writers')
    directors = ManyToManyField(DirectorModel, related_name='directors')

    rating = FloatField(null=True)
    url = CharField()
    description = CharField(null=True, max_length=10000)
    release_time = DateTimeField(null=True)
    year = IntegerField(null=True)
    month = IntegerField(null=True)
    day = IntegerField(null=True)

    class Meta:
        global db
        database = db  # This model uses the "people.db" database.


db.create_tables([GenreModel, CountryModel, ActorModel, DirectorModel, WriterModel, MovieModel,
                  MovieModel.genres.get_through_model(), MovieModel.countries.get_through_model(),
                  MovieModel.actors.get_through_model(), MovieModel.directors.get_through_model(),
                  MovieModel.writers.get_through_model()], safe=True)


def generate_actor_model(subject_id, name):
    try:
        model = ActorModel.get(
            subject_id=subject_id)
    except ActorModel.DoesNotExist:
        model = ActorModel.create(
            subject_id=subject_id, name=name)
        model.save()
    return model


def generate_director_model(subject_id, name):
    try:
        model = DirectorModel.get(
            subject_id=subject_id)
    except DirectorModel.DoesNotExist:
        model = DirectorModel.create(
            subject_id=subject_id, name=name)
        model.save()
    return model


def generate_writer_model(subject_id, name):
    try:
        model = WriterModel.get(
            subject_id=subject_id)
    except WriterModel.DoesNotExist:
        model = WriterModel.create(
            subject_id=subject_id, name=name)
        model.save()
    return model


def generate_genre_model(name):
    try:
        model = GenreModel.get(
            name=name)
    except GenreModel.DoesNotExist:
        model = GenreModel.create(
            name=name)
        model.save()
    return model


def generate_country_model(name):
    try:
        model = CountryModel.get(
            name=name)
    except CountryModel.DoesNotExist:
        model = CountryModel.create(
            name=name)
        model.save()
    return model


def is_movie_exist(subject_id):
    try:
        model = MovieModel.get(
            subject_id=subject_id)
        return True
    except MovieModel.DoesNotExist:
        return False

