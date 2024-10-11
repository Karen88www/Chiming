import openai
from articleFunc import *
from sheetFunc import *
from crawlFunc import *
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import requests
import json
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
import sqlite3
import os
import traceback
from typing import Union
from dotenv import load_dotenv

# dbPath = 'C:\\Users\\apuser\\code\\news-grouping-summary\\score.sqlite'
load_dotenv()
openai_api_key = os.getenv("Chiming_morning_KEY")
authFilePath = os.getenv("GOOGLE_CREDENTIALS_JSON")
fileId = os.getenv("File_ID")
# token = os.getenv("line_notify_token")
# dbPath = 'C:\\Users\\apuser\\code\\news-grouping-summary\\score.sqlite'

# 抓訊頭日期
def storyDtCnt(res):
    #因不同區域時差以發稿時間看可能會有出入，判斷文章發稿日期與訊頭日期，以做出新聞事件的時間基準
    articleDt = datetime.strptime(res['ResultData']['MetaData']['DateCreated'], '%Y/%m/%d %H:%M')
    storyDay = re.search('\d{1,2}日', res['ResultData']['News']['Content'][:40])
    #若無訊頭日期，則以文章發布日為主        
    storyDay = storyDay[0].split('日')[0] if storyDay else f"{articleDt:%d}"
    if storyDay == str(articleDt.day):
        return articleDt
    else:
        p = int(storyDay)- articleDt.day
        return articleDt + timedelta(days=p)

# 日期代換成yymmdd
def dateNounConverter(storyDt, text):
    #基於新聞事件時間基準，將日期相關名詞轉換為具體日期，可以持續擴增
    dateNoun = {
        '前天': f"前天（{storyDt+timedelta(days=-2):%Y年%m月%d日}）",
        '昨天': f"昨天（{storyDt+timedelta(days=-1):%Y年%m月%d日}）",
        '今天': f"今天（{storyDt:%Y年%m月%d日}）",
        '明天': f"明天（{storyDt+timedelta(days=1):%Y年%m月%d日}）",
        '明後天': f"明後天（{storyDt+timedelta(days=1):%Y年%m月%d日}"+"、"+f"{storyDt+timedelta(days=2):%Y年%m月%d日}）",
        '本月': f"本月{storyDt:%m月}",
        '上個月': f"上個月（{storyDt + relativedelta(months=-1):%m月}）",
        '下個月': f"下個月（{storyDt + relativedelta(months=+1):%m月}）",
        '今年': f"今年（{storyDt:%Y年}）",
        '去年': f"去年（{storyDt + relativedelta(years=-1):%Y年}）",
        '明年': f"明年（{storyDt + relativedelta(years=1):%Y年}）",
        }
    for key, value in dateNoun.items():
        text = text.replace(key, value)
    return text

# # 用getArticle拿文章、清理資料存成data
# def get_articles(done, ago=30):
#     # from_rss = newsInRss(f"https://www.cna.com.tw/googlenewssitemap_fromremote_cfp.xml?v={int(time.time())}")
#     from_rss = xml_news(f"https://www.cna.com.tw/googlenewssitemap_fromremote_cfp.xml?v={int(time.time())}")
#     #from_rss = getListNews()
#     pattern_today = f"{datetime.today():%Y%m%d}"+"\d{4}"
#     thirty_mins_ago = datetime.now() - timedelta(minutes=ago)
#     pids = [re.search('\d{12}', i['url']).group() for i in from_rss if bool(re.search(pattern_today, i['url'])) and datetime.strptime(i['dt'], "%Y/%m/%d %H:%M") < thirty_mins_ago and re.search('\d{12}', i['url']).group() not in done]
#     strip_info = '(（中央社[\w／、]{4,35}(專*電|報導|稿)）|[（(][\w／\/:： ]*[）)]\d{0,8}$)'
#     data = []
#     for pid in pids:
#         try:
#             res = getArticle(pid)
#             storyDt = storyDtCnt(res)
#             #清理新聞內容中的格式、程式註記
#             paragraphs = re.sub(strip_info,'',BeautifulSoup(res['ResultData']['News']['Content'].replace('(*#*)','')).text)
#             paragraphs = dateNounConverter(storyDt, paragraphs)
#             d = {
#                 "title": f"{res['ResultData']['News']['Keywords']} {res['ResultData']['News']['Title']}".strip(), 
#                 'content': [paragraphs], 
#                 'editor': [], 
#                 'keywords': res['ResultData']['News']['Keywords'], 
#                 'h1': res['ResultData']['News']['Title'], 
#                 'dt': res['ResultData']['MetaData']['DateModified'], 
#                 'attachments': len(res['ResultData']['News']['Photos']), 
#                 'extender': sum([len(c['Photo'].split('|$@$|')) for c in res['ResultData']['News']['Photos'] if c['iType']=='extender']),
#                 'category': res['ResultData']['News']['TypeName']}
#             data.append(d)
#             done.append(pid)
#         except:
#             continue
    
#     return data, pids, done


def get_focus():
    #取得網路組編輯將新聞選上重要版面的資料，原本有使用後端API，但因為API不穩定停用，改為爬取網頁

    # backend_feed = 'https://www.cna.com.tw/cna2018api/api/WNewsList/'
    # focus = []
    # cnt = 0
    # while cnt < 3:
    #     try:
    #         res = requests.post(backend_feed, data={'action': '0', 'category': 'headlines'}, headers={"Content-Type": "application/json", 'User-Agent': 'CNA-crawler-wrfAYr4ZaAXyaRu'}).json()
    #         focus = [i['Id'] for i in res['ResultData']['Items']]
    #         break
    #     except TypeError:
    #         return focus
    #     except:
    #         print('retrying...')
    #         time.sleep(2)
    #         focus = get_focus()
    #         cnt += 1
    
    url = 'https://www.cna.com.tw/list/headlines.aspx'
    soup = BeautifulSoup(requests.get(url, headers={"Cache-Control": "no-cache","Pragma": "no-cache", 'User-Agent': 'CNA-crawler-wrfAYr4ZaAXyaRu'}).text, 'lxml').find('div', {'class': 'statement'}).find_all('li')
    focus = [re.search('\d+\.aspx', li.find('a')['href'])[0][0:-5] for li in soup if '/news/' in str(li)]
    return focus

# # google rends的結果存在本機
# def get_gTrends(dbPath, date_start):
#     #取得google trends的關鍵字<另一支爬蟲固定更新於sqlite，以做為新聞分類的參考
#     with sqlite3.connect(dbPath) as db:
#         db.text_factory = str
#         keywords = pd.read_sql(f"select title, related from gTrends where dt>='{date_start}'", con=db)
#     res = []
#     # split keywords['related'] by str: ', '
#     keywords['related'] = keywords['related'].apply(lambda x: x.split(', '))
#     # combine keywords['title'] and keywords['related'] into the res list 
#     for i in range(len(keywords)):
#         res.append(keywords['title'][i])
#         res + keywords['related'][i]
#     return res


def single_stage_dbscan(data, distance_matrix, initial_eps, min_samples):
    #單一梯次計算相似聚類
    final_clusters = pd.DataFrame(columns=data.columns)
    dbscan = DBSCAN(eps=initial_eps, min_samples=min_samples, metric='precomputed')
    labels_first_stage = dbscan.fit_predict(distance_matrix)
    data['cluster_first_stage'] = labels_first_stage
    
    unique_clusters_first_stage = set(labels_first_stage)
    for cluster in unique_clusters_first_stage:
        if cluster == -1:  # -1 label represents noise in DBSCAN
            noise_points = data[data['cluster_first_stage'] == -1].copy()
            final_clusters = pd.concat([final_clusters, data], ignore_index=True)
            continue
    
    return final_clusters


def iterative_dbscan(data, distance_matrix, initial_eps, min_samples):
    #多階層計算聚類，但效果差距不大，目前停用
    final_clusters = pd.DataFrame(columns=data.columns)
    dbscan = DBSCAN(eps=initial_eps, min_samples=min_samples, metric='precomputed')
    labels_first_stage = dbscan.fit_predict(distance_matrix)
    data['cluster_first_stage'] = labels_first_stage
    
    unique_clusters_first_stage = set(labels_first_stage)
    for cluster in unique_clusters_first_stage:
        if cluster == -1:  # -1 label represents noise in DBSCAN
            noise_points = data[data['cluster_first_stage'] == -1].copy()
            final_clusters = final_clusters.append(noise_points)
            continue
        
        sub_data = data[data['cluster_first_stage'] == cluster].copy()
        if len(sub_data) <= 1:
            final_clusters = final_clusters.append(sub_data)
            continue
        
        reduced_eps = initial_eps
        min_samples_2nd = 2
        sub_distance_matrix = distance_matrix[sub_data.index][:, sub_data.index]
        dbscan = DBSCAN(eps=reduced_eps, min_samples=min_samples_2nd, metric='precomputed')
        sub_data['cluster_second_stage'] = dbscan.fit_predict(sub_distance_matrix)
        
        # Append noise points from the second stage
        noise_points_second_stage = sub_data[sub_data['cluster_second_stage'] == -1].copy()
        #final_clusters = final_clusters.append(noise_points_second_stage)
        final_clusters = final_clusters.append(sub_data)
    
    return final_clusters


def find_neighbors(df, threshold_high, threshold_low):
    #計算新聞間的相似度，這段函式主要是算出母體的距離標準(透過parameter輸入閥值範圍，經驗參考值)，提供eps供dbscan，最末有呼叫聚類函式
    embeddings = df['embeddings'].apply(np.array).tolist()
    # 計算距離矩陣
    distance_matrix = pdist(embeddings, metric='euclidean')
    distance_matrix = squareform(distance_matrix)
    # 使用 NearestNeighbors 找到每個點的最近鄰居
    neigh = NearestNeighbors(n_neighbors=2)  # 因為 min_samples = 2
    nbrs = neigh.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    distances = sorted(distances[:, 1], reverse=True) 
    # 計算距離的差分
    diffs = np.diff(distances)

    # 計算差分的絕對值，找到第一個顯著的轉折點
    diffs_abs = np.abs(diffs)
    distances_array = np.array(distances)
    # 找到threshold_high和threshold_low之間的差分的索引，500次嘗試
    cnt = 0
    while cnt < 500:
        try:
            indices_in_range = np.where((distances_array < threshold_high) & (distances_array > threshold_low))[0]
            turning_point_index_within_range = indices_in_range[np.argmax(diffs_abs[indices_in_range])]
            # 建議的 eps 值
            suggested_eps = distances_array[turning_point_index_within_range]
            print(suggested_eps)
            break
        except:
            threshold_low = threshold_low + 0.01
            print(threshold_low)
            cnt += 1
        # 指定的 eps 值
    if cnt >= 500 and suggested_eps > 1:
        suggested_eps = threshold_low * 2
        print("Reached maximum attempts. Using default eps value of 0.38")
    # 執行迭代的 DBSCAN 分析
    final_clusters = single_stage_dbscan(df, distance_matrix, suggested_eps, 2)
    return final_clusters


def df_tagging(df, focus=True, keywords='keywords today'): # 暫時刪掉 gTrends=-3
    #透過是否上重要版面、是否包含熱搜字、是否在編輯指定關鍵字進行貼標，並且透過score_sorting函式計算分數與排序
    df = df[~df['article'].str.contains(r'早安世|一週大事|開箱老照|(今|雙贏|威力)彩|大樂透|三大法人|新台幣.盤|新台幣[升貶].*收|(電子|金融|台股)期|台股[開收].*點|^台股[漲跌].{2,7}點$', regex=True)]

    try:
        df['embeddings'] = [json.loads(i) for i in df['embeddings']]
    except:
        pass
    
    focus = get_focus()
    df['focus'] = df['pid'].astype(str).apply(lambda x: 1 if x in focus else 0)

    # #權重: 是否符合google trends
    # gTrends = get_gTrends(dbPath, f"{datetime.today()+timedelta(days=-3):%Y/%m/%d}")
    # match = lambda x: len([bool(re.search(i, x)) for i in gTrends if bool(re.search(i, x)) ==True])
    # df['gTrends'] = df['text'].apply(match)


    #權重: 是否在鎖定關鍵字
    keywords_list = "" if get_cell(authFilePath, fileId, keywords, "B1") == None else get_cell(authFilePath, fileId, keywords, "B1")
    keywords_list = keywords_list.split('：') if '：' in keywords_list else keywords_list

    tag = lambda x: '|'.join(sorted([i for i in keywords_list if bool(re.search(i, x))]))
    tag_score = lambda x: sum([len(re.findall(i, x)) for i in keywords_list])
    df['inKeywords'] = df['h1'].apply(tag_score) * 2 + df['text'].apply(tag_score)
    df['keywordsTag'] = df['text'].apply(tag)
    
    df = score_sorting(find_neighbors(df, 0.8, 0.5))

    #根據分組排序、分數排序，回傳組名
    df1 = df.loc[df['cluster_first_stage'] > -1].sort_values(by='sort_cri', ascending=False).reset_index()
    df_grouped = df1.groupby('cluster_first_stage', as_index=False)['sort_cri'].sum()
    df_grouped['original_sort_cri'] = df1.groupby('cluster_first_stage')['sort_cri'].first().values
    df_sorted = df_grouped.sort_values(by=['sort_cri', 'original_sort_cri'], ascending=[False, False])
    threshold = df_sorted['sort_cri'].quantile(0.2)
    group = df_sorted[df_sorted['sort_cri'] > threshold]['cluster_first_stage'].tolist()

    return df, group


def score_sorting(df):
    #計算每篇的文章分數 1.發稿時間新到舊取對數、2.熱搜字、3.是否在編輯指定關鍵字、4.是否在google trends、5.附件數量
    df['position_score'] = np.log1p(df.index)

    df['sort_cri'] = df['inKeywords'] * 0.2 + df['focus'] * 0.4 + df['extender'] * 0.15 + df['attachments'] * 0.15 + df['position_score'] * 0.05 # 暫時刪掉+ df['gTrends'] * 0.2 

    df = df.sort_values(by='sort_cri', ascending=False).reset_index()
    return df


def combineUrl(x: Union[list, pd.Series]):
    #依據分類回推文章url
    if isinstance(x, list):
        category = x[0]
        pid = x[1]
    elif isinstance(x, pd.Series):
        category = x['category']
        pid = x['pid']
    cate = {'政治': 'aipl', '國際': 'aopl', '兩岸': 'acn', '產經': 'aie', '證券': 'asc','科技': 'ait', '生活': 'ahel', '社會': 'asoc', '地方': 'aloc', '文化': 'acul', '運動': 'aspt', '娛樂': 'amov'}
    return f"{cate.get(category)}/{pid}"


def get_related(df, df_before, done):
    #計算相關文章
    try:
        df_all = pd.concat([df, df_before]).reset_index(drop=True)
        df_all['pid'] = df_all['pid'].astype(str)
        df_all.sort_values(['pid'], ascending=False)[:10000]
        if 'related' not in df.columns:
            df['related'] = ''
        to_do = df.loc[df['pid'].astype(str).isin(done) == False]
        for n,i in enumerate(to_do.index):
            temp = {}
            e = to_do.loc[i, 'embeddings']
            for ind, j in enumerate(df_all['embeddings']):
                k = combineUrl(df_all.loc[ind])
                similarity = cosine_similarity([e], [j])[0][0]
                temp[k] = similarity if similarity > 0.5 else 0
            nearest_3 = sorted(temp, key=temp.get, reverse=True)[1:4]
            to_do.at[i, 'related'] = str(nearest_3)
            if n % 9 == 0:
                print(f"{n+1} articles found relations.")
        return to_do
    except:
        trace = traceback.format_exc()
        ind = re.search('line \d*\,', trace).span()[0]
        print(f"relation失敗，錯誤訊息：{trace[ind:ind+1500]}")
        # notify(f"relation失敗，錯誤訊息：{trace[ind:ind+1500]}", token)
        return df
    

# set open ai
# OpenAI API 的設置
def get_completion(messages, model="gpt-4o", temperature=0):
    payload = { "model": model, "temperature": temperature, "messages": messages}
    headers = { "Authorization": f'Bearer {openai_api_key}', "Content-Type": "application/json" }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers = headers, data = json.dumps(payload) )
    obj = json.loads(response.text)
    if response.status_code == 200:
        return obj["choices"][0]["message"]["content"]
    else:
        return obj["error"]

# 寫metadata
# for generate metadata
def generate_metadata(summarise, article):
    full_prompt = summarise + f"```{article}```\n"
    messages = [
        { "role": "user", "content": full_prompt }
    ]
    return get_completion(messages)

# 結構化新聞
def metaData(df):
    # prompt: what happen?
    whatHappen50 = """
    你 大語言模型 語料中央社新聞報導 熟悉中央社新聞報導語法和用字遣詞
    任務 閱讀文章寫100字摘要

    1. 實體辨識 取文本內實體Entity 以人物(不包含中央社記者)、組織、事件、地名、日期、關係分組 分別條列實體
    2. 判斷新聞核心事件或議題 包含但不限 事件名稱、相關人物、關鍵事件、日期時間
    3. 寫事件摘要：50個字以內、符合新聞書寫格式

    人名第一次提到 如果是翻譯 格式: 中文人名（原文） 用全形括弧 不是翻譯就不用
    提及人物 有職稱 格式: 人名(職稱) 用全形括弧

    注意 最終只要印出事件摘要結果
    注意！正確性最重要 要確保實體關係是根據文本 純文字不用markdow
    請注意！這對我的工作很重要 鎖定在我提供資料 要遵守
    """

    whatHappen200 = """
    你 大語言模型 語料中央社新聞報導 熟悉中央社新聞報導語法和用字遣詞
    任務 閱讀文章寫100字摘要

    1. 實體辨識 取文本內實體Entity 以人物(不包含中央社記者)、組織、事件、地名、日期、關係分組 分別條列實體
    2. 判斷新聞核心事件或議題 包含但不限 事件名稱、相關人物、關鍵事件、日期時間
    3. 寫事件摘要：150-200個字以內、符合新聞書寫格式

    人名第一次提到 如果是翻譯 格式: 中文人名（原文） 用全形括弧 不是翻譯就不用
    提及人物 有職稱 格式: 人名(職稱) 用全形括弧

    注意 最終只要印出事件摘要結果
    注意！正確性最重要 要確保實體關係是根據文本 純文字不用markdow
    請注意！這對我的工作很重要 鎖定在我提供資料 要遵守
    """

    # prompt: what are the key facts?
    keyFacts = """
    你 大語言模型 語料中央社新聞報導 熟悉中央社新聞報導語法和用字遣詞
    任務 閱讀文章分條列出關鍵事件

    1. 實體辨識 取文本內實體Entity 以人物(不包含中央社記者)、組織、事件、地名、日期、關係分組 分別條列實體
    2. 判斷關鍵事件 目的要凸顯事件發生經過，包含但不限：發生事件名稱 日期 地點 時間 
    3. 分條列出關鍵事件，不要超過5點。每一關鍵事件字數100個字以內

    時間日期很重要 格式 年月日 時：分

    注意 最終只要印出關鍵事件
    注意！正確性最重要 要確保實體關係是根據文本 純文字不用markdow
    請注意！這對我的工作很重要 鎖定在我提供資料 要遵守
    """

    # prompt: what key person say?
    stance = """
    你 大語言模型 語料中央社新聞報導 熟悉中央社新聞報導語法和用字遣詞
    任務 閱讀文章寫各方立場文

    1. 實體辨識 取文本內實體Entity 以人物(不包含中央社記者)、組織、事件、地名、日期、關係分組 分別條列實體
    2. 判斷各方立場文 目的在說明相關人或機構的立場、說法，包含但不限：相關人或機構名稱 說法、看法、立場
    3. 分段列出各方立場文 格式 機構或人名：100個字以內立場文

    人名第一次提到 如果是翻譯 格式: 中文人名（原文） 用全形括弧 不是翻譯就不用
    提及人物 有職稱 格式: 人名(職稱) 用全形括弧

    注意 最終只要印出各方立場文
    注意！正確性最重要 要確保實體關係是根據文本 純文字不用markdow
    請注意！這對我的工作很重要 鎖定在我提供資料 要遵守
    """

    if not df.empty:
        for index, row in df.iterrows():
            print(f"processing: {row['pid']}, \n {index+1}/{len(df)}")
            
            # 創建新的欄位
            what_happen_50 = generate_metadata(whatHappen50, row['article'])
            df.at[index, 'whatHappen50'] = what_happen_50
            print(f"whatHappen50 for pid {row['pid']}: {what_happen_50}")

            df.at[index, 'whatHappen200'] = generate_metadata(whatHappen200, row['article'])
            df.at[index, 'keyFacts'] = generate_metadata(keyFacts, row['article'])
            df.at[index, 'stance'] = generate_metadata(stance, row['article'])

        meta_file = f'metadata/metaData-{datetime.today():%Y%m%d}.csv'
        df.to_csv(meta_file, sep='\t', encoding='utf-8', index=False)
        return df
    else:
        print(f"No data to process.")
        return pd.DataFrame()
    

# for generate summarise
def generate_morning(summarise, df):
    # 收集所有 row 的摘要資料
    combined_summaries = ""

    for index, row in df.iterrows():
        combined_summaries += f"""
        事件摘要: {row['whatHappen']}
        關鍵事件: {row['keyFacts']}
        立場文: {row['stance']}
        """

    full_prompt = summarise + combined_summaries

    messages = [
        { "role": "user", "content": full_prompt }
    ]
    
    return get_completion(messages)
