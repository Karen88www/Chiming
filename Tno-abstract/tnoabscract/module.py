import re
import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('Chiming_abstract_KEY')


# 提取 tno 函數
def extract_tno(url):
    pattern = r'/topic/newstopic/(\d+)\.aspx'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None

# 根據 tno 獲取所有文章 id
def getTopic(tno):
    data = json.dumps({"tno": str(tno), "action": "0"})
    headers = {"Content-Type": "application/json", 'User-Agent': 'CNA-crawler-wrfAYr4ZaAXyaRu'}
    res = requests.post('https://www.cna.com.tw/cna2018api/api/ProjTopic', headers=headers, data=data).json()
    return res

# 設定錯誤判斷: 當tno不存在
class DataStructureError(Exception):
    pass

def get_title_data(title_data):
    if 'ResultData' in title_data and 'Items' in title_data['ResultData']:
        return pd.DataFrame([
            {'pid': i['Id'], 'published_dt': i['CreateTime'], 'h1': i['HeadLine']}
            for i in title_data['ResultData']['Items']
        ])
    else:
        print("Error: Unexpected data structure from getTopic")
        raise DataStructureError("Error: Unexpected data structure from getTopic")

# 根據 pid 獲取文章數據
def getArticle(pid):
    data = json.dumps({"id": str(pid)})
    headers = {"Content-Type": "application/json", 'User-Agent': 'CNA-crawler-wrfAYr4ZaAXyaRu'}
    res = requests.post('https://www.cna.com.tw/cna2018api/api/ProjNews', headers=headers, data=data).json()
    return res


# 計算 extender
def extender_counter(res):
    if 'ResultData' in res and 'News' in res['ResultData']:
        return len([url for photo in res['ResultData']['News']['Photos']
                    if photo.get('iType') == 'extender'
                    for url in photo.get('Photo', '').split('|$@$|')])
    return 0

# 計算文章日

# check date
def story_dt_cnt(res):
    # 發稿日期
    articleDt = datetime.strptime(res['ResultData']['MetaData']['DateCreated'], '%Y/%m/%d %H:%M')
    # 訊頭日期
    storyDay = re.search('\d{1,2}日', res['ResultData']['News']['Content'][:40])
    storyDay = storyDay[0].split('日')[0] if storyDay else f"{articleDt:%d}"   
    # 計算日期差距，換成新的sotryDay
    p = int(storyDay)- articleDt.day
    storyDay = articleDt + timedelta(days=p)
    # print(storyDay) # 10/30
    
    # 比較storyDay和articleDt,檢查有沒有跨月
    if storyDay > articleDt:
        storyDay = storyDay - relativedelta(months=1)
        print(">> sotryDay:", storyDay)
        return storyDay
    else:
        print(">> sotryDay:", storyDay)
        return storyDay
    

# 轉換日期名詞
# def date_noun_converter(story_dt,text):
#     dateNoun = {
#             '前天': f"前天（{story_dt+timedelta(days=-2):%Y年%m月%d日}）",
#             '昨天': f"昨天（{story_dt+timedelta(days=-1):%Y年%m月%d日}）",
#             '今天': f"今天（{story_dt:%Y年%m月%d日}）",
#             '明天': f"明天（{story_dt+timedelta(days=1):%Y年%m月%d日}）",
#             '明後天': f"明後天（{story_dt+timedelta(days=1):%Y年%m月%d日}"+"、"+f"{story_dt+timedelta(days=2):%Y年%m月%d日}）",
#             '本月': f"本月{story_dt:%m月}",
#             '上個月': f"上個月（{story_dt + relativedelta(months=-1):%m月}）",
#             '下個月': f"下個月（{story_dt + relativedelta(months=+1):%m月}）",
#             '今年': f"今年（{story_dt:%Y年}）",
#             '去年': f"去年（{story_dt + relativedelta(years=-1):%Y年}）",
#             '明年': f"明年（{story_dt + relativedelta(years=1):%Y年}）"
#             }
    
#     for key, value in dateNoun.items():
#         text = text.replace(key, value)
#     print(">>>置換日期之後的結果\n",  text)
#     return text

def date_noun_converter(story_dt, paragraphs):
    prompt = """
    你 大語言模型 語料 中央社新聞報導
    任務 以發稿日期(df['dt'])為標準，根據以下步驟，轉換文章中提到的今天 明天 後天 某日等日期說法

    ## 以下是偽代碼
    step 1.  以storyDay的日期為基準，置換內文日期
    storyDay = story_dt_cnt(res)
    
    text = df['article]

    置換的邏輯如下，只要是描述日期的都請置換成datetime格式
    dateNoun = {
            '前天': f"前天（{story_dt+timedelta(days=-2):%Y年%m月%d日}）",
            '昨天': f"昨天（{story_dt+timedelta(days=-1):%Y年%m月%d日}）",
            '今天': f"今天（{story_dt:%Y年%m月%d日}）",
            '明天': f"明天（{story_dt+timedelta(days=1):%Y年%m月%d日}）",
            '明後天': f"明後天（{story_dt+timedelta(days=1):%Y年%m月%d日}"+"、"+f"{story_dt+timedelta(days=2):%Y年%m月%d日}）",
            '本月': f"本月{story_dt:%m月}",
            '上個月': f"上個月（{story_dt + relativedelta(months=-1):%m月}）",
            '下個月': f"下個月（{story_dt + relativedelta(months=+1):%m月}）",
            '今年': f"今年（{story_dt:%Y年}）",
            '去年': f"去年（{story_dt + relativedelta(years=-1):%Y年}）",
            '明年': f"明年（{story_dt + relativedelta(years=1):%Y年}）"
            }

    特別注意！如果有出現類似 2日的寫法，也要置換： '某日': f"某日（{story_dt+timedelta(days=某日跟sotryDay的差值):%Y年%m月%d日}）"
    如果有出現類似 11月15日 的寫法，也要置換： '某月某日': f"某月某日（{story_dt + relativedelta(year=storyDay.year, months=某月跟storyDay的差值, days=某日跟storyDay的差值):%Y年%m月%d日}）"

    step 3. 置換完成後，請再次比對published_dt 確認日期的年 月 日判斷都合理

    step 4. 
    for key, value in dateNoun.items():
        text = text.replace(key, value)
        return text

    日期置換格式遵守 如：今天（2024年10月15日） 今年（2024年） 本月9日（2024年10月09日）
    注意！不要更動article的原文或擅自加字 原文中的今天 明天 昨天 本月9日 等類似字樣必須保留

    最後只要print text 純文字不用markdown
"""
    storyDay = story_dt
    article = story_dt, paragraphs
    article_c = generate_date_convert(prompt, article, storyDay)
    print(f'>>> storyDay:{storyDay} \n>>>置換後文章: \n{article_c}')

    return article_c

# 清理文章內容
def clean_text(res):

    strip_info = r'[（(][\w／\/:： ]*[）)]\d{0,8}$'
    story_dt = story_dt_cnt(res)
    news_content = res['ResultData']['News']['Content']
    # print("news content \n", news_content)
    paragraphs = re.sub(strip_info, '', BeautifulSoup(news_content.replace('(*#*)', ''), 'html.parser').text)
    print(">>> 清理後的結果\n", paragraphs)
    return date_noun_converter(story_dt, paragraphs)


# 保存文章數據
def get_article_data(pid):
    article_data = getArticle(pid)
    news = article_data['ResultData']['News']
    return {
        'pid': news['Id'],
        'published_dt': article_data['ResultData']['MetaData']['DateCreated'],
        'h1': news['Title'],
        'article': clean_text(article_data),
        'extender': extender_counter(article_data),
        'category': news['TypeName']
    }


def process_articles(title_df):
    article_data_list = []
    for pid in title_df['pid']:
        article_data_list.append(get_article_data(pid))
        time.sleep(2)
    return pd.DataFrame(article_data_list)

# 計算加權分數
def calculate_weighted_scores(df):
    df['ex_score'] = df['extender'] * 1.5
    date_counts = df['published_dt'].value_counts().sort_values(ascending=False)
    date_weights = {date: len(date_counts) - i for i, date in enumerate(date_counts.index)}
    df['weighted_score'] = df.apply(lambda row: row['ex_score'] + date_weights[row['published_dt']], axis=1)
    df = df.sort_values('weighted_score', ascending=False)
    df_filtered = df[df['weighted_score'] > df['weighted_score'].max() / 2]
    return df_filtered


# OpenAI API 的設置
def get_completion(messages, model="gpt-4o", temperature=0):
    payload = { "model": model, "temperature": temperature, "messages": messages}
    headers = { "Authorization": f'Bearer {openai_api_key}', "Content-Type": "application/json" }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers = headers, data = json.dumps(payload) )
    obj = json.loads(response.text)
    if response.status_code == 200:
        return obj["choices"][0]["message"]["content"]
    else:
        print(obj["error"])
        return obj["error"]

# 生成日期轉換內容
def generate_date_convert(prompt, data, story_dt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"請根據 story_dt: {story_dt} 轉換以下文本中的日期表述：\n\n{data}"}
    ]
    return get_completion(messages)


# 生成摘要內容
def generate_content(prompt, data, colName):
    full_prompt = prompt
    for index, row in data.iterrows():
        full_prompt += f"```{row[colName]}```\n"
    messages = [
        { "role": "user", "content": full_prompt }
    ]
    return get_completion(messages)

# 生成module
def generate_metadata(prompt, data):
    full_prompt = prompt
    full_prompt += f"```{data['article']}```\n"
    
    messages = [
        { "role": "user", "content": full_prompt }
    ]
    return get_completion(messages)

## 結構化資料
def metaData(tno):
    # prompt: what happen?
    whatHappened50 = """
    你 大語言模型 語料中央社新聞報導 熟悉中央社新聞報導語法和用字遣詞
    任務 閱讀文章寫100字摘要

    1. 實體辨識 取文本內實體Entity 以人物(不包含中央社記者)、組織、事件、地名、日期、關係分組 分別條列實體
    2. 判斷新聞核心事件或議題 包含但不限 事件名稱、相關人物、關鍵事件、日期時間
    3. 寫事件摘要：50個字以內、符合新聞書寫格式

    人名第一次提到 如果是翻譯 格式: 中文人名（原文） 用全形括弧 不是翻譯就不用
    提及人物 有職稱 格式: 人名(職稱) 用全形括弧

    注意 最終只要印出事件摘要結果
    注意！正確性最重要 要確保實體關係是根據文本 純文字不用markdown
    請注意！這對我的工作很重要 鎖定在我提供資料 要遵守
    """

    whatHappened200 = """
    你 大語言模型 語料中央社新聞報導 熟悉中央社新聞報導語法和用字遣詞
    任務 閱讀文章寫100字摘要

    1. 實體辨識 取文本內實體Entity 以人物(不包含中央社記者)、組織、事件、地名、日期、關係分組 分別條列實體
    2. 判斷新聞核心事件或議題 包含但不限 事件名稱、相關人物、關鍵事件、日期時間
    3. 寫事件摘要：150-200個字以內、符合新聞書寫格式

    人名第一次提到 如果是翻譯 格式: 中文人名（原文） 用全形括弧 不是翻譯就不用
    提及人物 有職稱 格式: 人名(職稱) 用全形括弧

    注意 最終只要印出事件摘要結果
    注意！正確性最重要 要確保實體關係是根據文本 純文字不用markdown
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
    注意！正確性最重要 要確保實體關係是根據文本 純文字不用markdown
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
    注意！正確性最重要 要確保實體關係是根據文本 純文字不用markdown
    請注意！這對我的工作很重要 鎖定在我提供資料 要遵守
    """

    num, title_df = count_num(tno)
    print(f"tno: {tno}, article_num : {num}")

    if not title_df.empty:
        df = process_articles(title_df)

        # 生成摘要的DataFrame
        summary_list = []
        print("-----生成metadata-----")
        for index, row in df.iterrows():
            print(f"pid: {row['pid']}")
            print(f'progress: {index+1}/{num}')
            res_happened50 = generate_metadata(whatHappened50, row)
            print(res_happened50)
            res_happened200 = generate_metadata(whatHappened200, row)
            # print(res_happened200)
            res_keyFact = generate_metadata(keyFacts, row) 
            res_stance = generate_metadata(stance, row)

            # 將摘要和關鍵事件添加到summary_list
            summary_list.append({
                "tno": int(tno),
                "pid": row['pid'],
                "whatHappened50": res_happened50,
                "whatHappened200" : res_happened200,
                "keyFacts": res_keyFact,
                "stance": res_stance
            })
        
        # 將摘要數據轉換為 DataFrame
        summaries_df = pd.DataFrame(summary_list)
        if summaries_df is not None:
            print("Metadata saved successfully")
        else:
            print("Metadata fail to save")
        
        # 返回包含摘要的DataFrame
        return summaries_df
    else:
        print(f"No data to process for tno: {tno}")
        return pd.DataFrame()
    
def count_num(tno):
    title_data = getTopic(tno)
    num = len(title_data['ResultData']['Items'])
    title_df = get_title_data(title_data)
    return num, title_df

