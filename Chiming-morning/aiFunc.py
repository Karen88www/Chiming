import requests
import time
import backoff 
import openai
import json
import re
import os
import tiktoken


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chatWithBackOff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def cut_before_embedding(data:dict) -> str:
    # data:{"title": <str>, "content": [<str>, <str>, ...], "editor": [<str>, <str>, ...]}
    headers = {"Content-Type": "application/json"}
    url = 'https://ckiptagger.cna.com.tw:8881/origin/'
    #url = 'http://127.0.0.1:8889/origin/'
    res = requests.post(url, data=json.dumps(data), headers=headers).json()
    if res['Result'] == 'Y':
        cut = [y for x in res['ResultData']['cut'] for y in x]
        pos = [y for x in res['ResultData']['pos'] for y in x]
        entity = [y for x in res['ResultData']['entity'] for y in x]
        text = pos_filter(cut, pos, entity)
    return data['title'] + ' ' + text


def pos_filter(cut, pos, entity):
    mark = ["COLONCATEGORY","COMMACATEGORY","DASHCATEGORY","DOTCATEGORY","ETCCATEGORY","EXCLAMATIONCATEGORY","PARENTHESISCATEGORY","PAUSECATEGORY","PERIODCATEGORY","QUESTIONCATEGORY","SEMICOLONCATEGORY","SPCHANGECATEGORY","WHITESPACE"]
    text = ''
    for c, p in zip(cut, pos):
        if bool(re.search('^N|^V|Cbb|Caa|^D$', p)):
            text += c
        elif p in mark:
            text += ' '
    e = sorted(set([ent[0] for ent in entity if ent[1] in ['ORG', 'GPE', 'DATE', 'PERSON','EVENT', 'FAC', 'WORK_OF_ART', 'LAW','LOC'] and len(ent[0]) > 1]))
    e_text = ' '.join(e)
    return text + ' entities: '+ e_text


def text_embeddings_3(text):
    t = openai.Embedding.create(
    model="text-embedding-3-large",
    input=text
    )
    return t['data'][0]['embedding']


def text_embeddings(text):
    t = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=text
    )
    return t['data'][0]['embedding']


def msgTokenCnt(msg):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(msg))
    return num_tokens