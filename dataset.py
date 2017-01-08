# -*- coding: utf-8 -*-

from peewee import *
import datetime
import numpy as np
import matplotlib.pyplot as plt
from playhouse.fields import ManyToManyField
import util
import os
from orm import MovieModel
from orm import CountryModel

# >>> orm.MovieModel.select().wrapped_count()
# 26588L
# >>> orm.ActorModel.select().wrapped_count()
# 62632L
# >>> orm.GenreModel.select().wrapped_count()
# 48L
# >>> orm.CountryModel.select().wrapped_count()
# 307L
# >>> orm.DirectorModel.select().wrapped_count()
# 16804L
# >>> orm.WriterModel.select().wrapped_count()
# 21188L

GENRE_COUNT_INDEX = 0
COUNTRY_COUNT_INDEX = 1
ACTOR_COUNT_INDEX = 2
DIRECTOR_COUNT_INDEX = 3
WRITER_COUNT_INDEX = 4
MONTH_INDEX = 5

GENRE_INDEX = 100
COUNTRY_INDEX = 200
# end at 1000
ACTOR_INDEX = 1000
WRITER_INDEX = 70000
DIRECTOR_INDEX = 95000

TOTAL_LENGTH = 115000

index_array = np.chararray(TOTAL_LENGTH)


def fill_with_array_with_order(model_array, start_index, feature_row):
    count = len(model_array)
    for x in range(0, count):
        model = model_array[x]
        index = start_index + model.id
        if x == 0:
            value = 10
        elif x == 1:
            value = 5
        else:
            value = 1
        feature_row[index] = value
        global index_array
        index_array[index] = model.name


def fill_with_array(model_array, start_index, feature_row):
    for model in model_array:
        index = start_index + model.id
        feature_row[index] = 1
        global index_array
        index_array[index] = model.name


def generate_row(movie):
    global index_array
    feature_row = np.zeros(TOTAL_LENGTH)
    feature_row[GENRE_COUNT_INDEX] = len(movie.genres)
    index_array[GENRE_COUNT_INDEX] = "genre count"
    fill_with_array(movie.genres, GENRE_INDEX, feature_row)

    feature_row[COUNTRY_COUNT_INDEX] = len(movie.countries)
    fill_with_array(movie.countries, COUNTRY_INDEX, feature_row)
    index_array[COUNTRY_COUNT_INDEX] = "country count"

    feature_row[ACTOR_COUNT_INDEX] = len(movie.actors)
    fill_with_array_with_order(movie.actors, ACTOR_INDEX, feature_row)
    index_array[ACTOR_COUNT_INDEX] = "actor count"

    feature_row[DIRECTOR_COUNT_INDEX] = len(movie.directors)
    fill_with_array_with_order(movie.directors, DIRECTOR_INDEX, feature_row)
    index_array[DIRECTOR_COUNT_INDEX] = "director count"

    feature_row[WRITER_COUNT_INDEX] = len(movie.writers)
    fill_with_array_with_order(movie.writers, WRITER_INDEX, feature_row)

    month = movie.month
    month_index = MONTH_INDEX + (month - 1)
    feature_row[month_index] = 1
    index_array[MONTH_INDEX] = "month start"
    return feature_row


# basic rules:
# remove movies older than 1980
# only consider chinese movie
# remove movies with zero director/writer/actor/genres
# remove movie without rating

def get_models():
    # query_country = CountryModel.select().where(CountryModel.name.contains("\u4e2d\u56fd".encode("unicode-escape")))
    # for country in query_country:
    #     name = country.name
    #     print name
    # print query_country.wrapped_count()
    query = MovieModel.select().where(
        MovieModel.year >= 1980 and MovieModel.rating > 0)
    print query.wrapped_count()
    # china = "\u4e2d\u56fd".encode("unicode-escape")
    china = "\u4e2d\u56fd"
    features = np.array([])
    labels = np.array([])
    n_dim = 0
    for movie in query:
        valid = False
        if not movie.month:
            continue
        if len(movie.actors) == 0 or len(movie.directors) == 0 or len(movie.writers) == 0:
            continue
        for country in movie.countries:
            if country.name and china in country.name or 'china' in country.name.lower():
                valid = True
        if valid:
            feature_row = generate_row(movie)
            print feature_row
            n_dim = len(feature_row)
            rating = movie.rating
            labels = np.append(labels, [rating])
            features = np.append(features, feature_row)

            # test code

    features = features.reshape((len(features) / n_dim), n_dim)
    f_feature = open("all_feature.npz", 'w')
    f_label = open("all_label.npz", 'w')
    f_index = open("all_index.npz", 'w')
    np.save(f_feature, features)
    np.save(f_label, labels)
    np.save(f_index, index_array)


def extract_name_ids(array, func):
    result = []
    for name_id in array:
        items = name_id.split('|')
        if len(items) >= 2:
            result.append(func(items[0], items[1]))
    return result


def extract_names(array, func):
    result = []
    for name in array:
        result.append(func(name))
    return result


def fill_name_ids(feature_row, array, start_index, insert_func):
    for name_id in array:
        items = name_id.split('|')
        if len(items) >= 2:
            model = insert_func(items[0], items[1])
            index = int(model.id) + start_index
            feature_row[index] = 1


def fill_name_ids_1(feature_row, array, count, insert_func):
    for name_id in array:
        items = name_id.split('|')
        if len(items) >= 2:
            model = insert_func(items[0], items[1])
            feature_row.append(int(model.id))
            count -= 1
        if count <= 0:
            return
    while count > 0:
        feature_row.append(feature_row[len(feature_row) - 1])
        count -= 1


# def generate_row(movie):
#     feature_row = np.zeros(TOTAL_LENGTH)
#     genres = util.get_text(movie.genres).split(',')
#     feature_row[GENRE_COUNT_INDEX] = len(genres)
#     fill_name_ids(feature_row, genres, GENRE_INDEX, generate_genre)
#
#     countries = util.get_text(movie.countries).split(',')
#     feature_row[COUNTRY_COUNT_INDEX] = len(countries)
#     fill_name_ids(feature_row, countries, COUNTRY_INDEX, generate_country)
#
#     actors = util.get_text(movie.actors).split(',')
#     feature_row[ACTOR_COUNT_INDEX] = len(actors)
#     fill_name_ids(feature_row, actors, ACTOR_INDEX, generate_actor)
#
#     directors = util.get_text(movie.directors).split(',')
#     feature_row[DIRECTOR_COUNT_INDEX] = len(directors)
#     fill_name_ids(feature_row, directors, DIRECTOR_INDEX, generate_director)
#
#     writers = util.get_text(movie.writers).split(',')
#     feature_row[WRITER_COUNT_INDEX] = len(writers)
#     fill_name_ids(feature_row, writers, WRITER_INDEX, generate_writer)
#
#     # date 2016-12-16
#     date_str = util.get_text(movie.release_time)
#     date_items = date_str.split('-')
#
#     month = int(date_items[1])
#     month_index = MONTH_INDEX + (month - 1)
#     feature_row[month_index] = 1
#     return feature_row


# def generate_row(movie):
#     feature_row = np.zeros(TOTAL_LENGTH)
#     genres = util.get_text(movie.genres).split(',')
#     feature_row[GENRE_COUNT_INDEX] = len(genres)
#     fill_name_ids(feature_row, genres, GENRE_INDEX, generate_genre)
#
#     countries = util.get_text(movie.countries).split(',')
#     feature_row[COUNTRY_COUNT_INDEX] = len(countries)
#     fill_name_ids(feature_row, countries, COUNTRY_INDEX, generate_country)
#
#     actors = util.get_text(movie.actors).split(',')
#     feature_row[ACTOR_COUNT_INDEX] = len(actors)
#     fill_name_ids(feature_row, actors, ACTOR_INDEX, generate_actor)
#
#     directors = util.get_text(movie.directors).split(',')
#     feature_row[DIRECTOR_COUNT_INDEX] = len(directors)
#     fill_name_ids(feature_row, directors, DIRECTOR_INDEX, generate_director)
#
#     writers = util.get_text(movie.writers).split(',')
#     feature_row[WRITER_COUNT_INDEX] = len(writers)
#     fill_name_ids(feature_row, writers, WRITER_INDEX, generate_writer)
#
#     # date 2016-12-16
#     date_str = util.get_text(movie.release_time)
#     date_items = date_str.split('-')
#
#     month = int(date_items[1])
#     month_index = MONTH_INDEX + (month - 1)
#     feature_row[month_index] = 1
#     return feature_row


feature_path = "feature_china.npz"
label_path = "label_china.npz"


def extract_features():
    if os.path.isfile(feature_path):
        f_feature = open(feature_path, 'r')
        f_label = open(label_path, 'r')
        # return np.load(f_feature), np.load(f_label)
    features = np.array([])
    labels = np.array([])
    # movies = MovieModel.select().where(MovieModel.countries.contains('中国'))
    movies = MovieModel.select()
    n_dim = 0
    for movie in movies:
        # date 2016-12-16
        date_str = util.get_text(movie.release_time)
        date_items = date_str.split('-')
        if len(date_items) < 2:
            continue
        year = int(date_items[0])
        if year <= 2008:
            continue
        rating = float(util.get_text(movie.rating))
        rating = int(rating * 10)
        if rating <= 0:
            continue

        feature_row = generate_row(movie)
        n_dim = len(feature_row)

        labels = np.append(labels, [rating])
        features = np.append(features, feature_row)

        # test code

    features = features.reshape((len(features) / n_dim), n_dim)
    f_feature = open(feature_path, 'w')
    f_label = open(label_path, 'w')
    # np.save(f_feature, features)
    # np.save(f_label, labels)

    return features, labels


get_models()
