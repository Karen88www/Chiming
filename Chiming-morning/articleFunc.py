import requests
from bs4 import BeautifulSoup
import json

# def notify(msg, token):
#     lineNotifyToken = token
#     headers = {"Authorization" : "Bearer " + lineNotifyToken, "Content-Type" : "application/x-www-form-urlencoded"}
#     payload = {'message': msg}
#     requests.post("https://notify-api.line.me/api/notify", headers = headers, params = payload)


def getTopicGroup(tno):
    data = json.dumps({"tgno" : str(tno), "action": "0"})
    headers = {"Content-Type": "application/json", 'User-Agent': 'CNA-crawler-wrfAYr4ZaAXyaRu'}
    res =  requests.post('https://www.cna.com.tw/cna2018api/api/ProjTopicGroup', headers=headers, data=data).json()
    return res


def getTopic(tno):
    data = json.dumps({"tno" : str(tno), "action": "0"})
    headers = {"Content-Type": "application/json", 'User-Agent': 'CNA-crawler-wrfAYr4ZaAXyaRu'}
    res =  requests.post('https://www.cna.com.tw/cna2018api/api/ProjTopic', headers=headers, data=data).json()
    return res


def getArticleEng(pid):
    data = json.dumps({"siteid": "cnaeng", "category": "news","id": str(pid)})
    headers = {"Content-Type": "application/json", 'User-Agent': 'CNA-crawler-wrfAYr4ZaAXyaRu'}
    res = requests.post('https://focustaiwan.tw/cna2019api/cna/FTNews/', headers=headers, data=data).json()
    return res


def getArticle(pid):
    data = json.dumps({"id": str(pid)})
    headers = {"Content-Type": "application/json", 'User-Agent': 'CNA-crawler-wrfAYr4ZaAXyaRu'}
    res = requests.post('https://www.cna.com.tw/cna2018api/api/ProjNews', headers=headers, data=data).json()
    return res


def getListNews():
    headers = {"Content-Type": "application/json", 'User-Agent': 'CNA-crawler-wrfAYr4ZaAXyaRu'}#'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.84 Safari/537.36'}
    url = 'https://www.cna.com.tw/list/aall.aspx'
    res = requests.get(url, headers=headers).text
    li = BeautifulSoup(res, 'html.parser').select('ul#jsMainList > li')
    news = [{'title': i.select('h2> span')[0].text.strip(), 'url': i.select('a')[0]['href'], 'dt': i.select('.date')[0].text} for i in li]
    return news


def bitly(url, token, utm):
    if '?' in url:
        longUrl = url+'&'
    else:
        longUrl = url+'?'
    if utm == 'FB':
        longUrl = longUrl+'utm_source=site.facebook&utm_medium=share&utm_campaign=fbuser'
    elif utm == 'LINE':
        longUrl = longUrl+'utm_source=LINE&utm_medium=share&utm_campaign=lineuser'
    elif utm == 'YOUTUBE':
        longUrl = longUrl+'utm_source=youtube&utm_medium=share&utm_campaign=youtube_meta'
    elif utm == 'IG':
        longUrl = longUrl+'utm_source=instagram&utm_medium=share'
    elif utm == '':
        pass

    headers = {"Authorization": "Bearer b3c2a499133ea637dbf1bcd9beb148c6ef053fad", "Content-Type": "application/json"}
    data = {
        "long_url": longUrl,
        "domain": "bit.ly",
        "group_guid": "o_6n3r3roci6"
    }
    try:
        res = requests.post("https://api-ssl.bitly.com/v4/shorten", headers=headers, data=json.dumps(data)).json()['link']
    except:
        res = longUrl

    return res

