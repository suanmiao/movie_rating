# encoding: utf-8
import scrapy
from scrapy.selector import Selector
from scrapy.http import HtmlResponse
import json
import re
import time
import os
import requests
from parsel import Selector

NUM_RE = re.compile(r"(\d+)")
DATE_RE = re.compile(r"[\d,-]{8,20}")
COUNTRIES_RE = re.compile(ur"制片国家/地区:</span> (.+?)<br>")


class FakeResponse:
    def __init__(self, text):
        self.sel = Selector(text=text)

    def xpath(self, query):
        return self.sel.xpath(query)


def extract_text(a):
    name = a.xpath("text()").extract()
    if name:
        name = name[0].encode('utf-8')
        return name
    return None


def extract_id_href(a):
    name = a.xpath("text()").extract()
    director_id = a.xpath("@href").re(NUM_RE)
    if name and director_id:
        name = name[0].encode('utf-8')
        item_id = director_id[0].encode('utf-8')
        return name + '|' + item_id
    return None


def get_text(text):
    return text.encode('utf-8')


def get_directors(response, obj):
    # //a[@rel='v:directedBy']
    directors = []
    for a in response.xpath("//a[@rel='v:directedBy']"):
        extracted = extract_id_href(a)
        if extracted:
            directors.append(extracted)
    if len(directors) > 0:
        director_str = ",".join(directors)
        obj['directors'] = director_str


def get_writers(response, obj):
    # //span[contains(., '编剧')]//a
    writers = []
    for a in response.xpath(u"//span[contains(., '编剧')]//a"):
        extracted = extract_id_href(a)
        if extracted:
            writers.append(extracted)
    writer_str = ",".join(writers)
    obj['writers'] = writer_str


def get_actors(response, obj):
    # //span[contains(., '主演')]//a
    actors = []
    for a in response.xpath("//a[@rel='v:starring']"):
        extracted = extract_id_href(a)
        if extracted:
            actors.append(extracted)
    if len(actors) > 0:
        actors_str = ",".join(actors)
        obj['actors'] = actors_str


def get_genres(response, obj):
    # ////span[@property='v:genre']
    genres = []
    for a in response.xpath("//span[@property='v:genre']"):
        extracted = extract_text(a)
        if extracted:
            genres.append(extracted)
    if len(genres) > 0:
        genres_str = ",".join(genres)
        obj['genres'] = genres_str


def get_countries(response, obj):
    S = "".join(response.xpath("//div[@id='info']").extract())
    M = COUNTRIES_RE.search(S)
    if M is not None:
        obj["countries"] = ','.join([country.strip() for country in M.group(1).split("/")])


def get_release_time(response, obj):
    # //span[@property='v:initialReleaseDate']
    release_time = response.xpath("//span[@property='v:initialReleaseDate']")
    if release_time:
        m = release_time.re(DATE_RE)
        if m:
            release_time = m[0]
            obj['release_time'] = release_time


def fetch_movie_body_by_id(movie_id):
    url = 'https://movie.douban.com/subject/' + str(movie_id)
    headers = {
        "User-Agent": "Baiduspider",
        "Accept-Language": "zh-CN,zh"
    }
    r = requests.get(url, headers=headers)
    content = r.text
    print "status code " + str(r.status_code) + " data length " + str(len(content))
    response = FakeResponse(content)
    obj = {}
    get_release_time(response, obj)
    get_genres(response, obj)
    get_countries(response, obj)
    get_actors(response, obj)
    get_directors(response, obj)
    get_writers(response, obj)
    get_release_time(response, obj)

    print obj
