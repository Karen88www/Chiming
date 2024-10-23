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
openai_api_key = os.getenv('Chiming_KEY')

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

### 日期置換 ###
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
    article_time = generate_date_convert(prompt, article, storyDay)
    print(f'>>> storyDay:{storyDay} \n>>>時間置換後文章: \n{article_time}')

    return article_time

### 人名置換 ###
# 從實體辨識裡面抓出人物、機關、事件
def ckipArticleCutter(article=None):
    # 檢查 article 是否返回有效結果
    if not isinstance(article, dict):
        # raise ValueError(f"錯誤：取得的文章不是有效的字典。")
        print(f"錯誤：取得的文章不是有效的字典。")
        return {}

    if 'ResultData' not in article or 'News' not in article['ResultData']:
        # raise ValueError(f"錯誤：找不到給定 'article_id' {article_id} 對應的文章。")
        print(f"錯誤：找不到給定 'article_id' {article_id} 對應的文章。")
        return {}
    
   # 處理文章內容
    title = article['ResultData']['News'].get('Title', '')
    content = [BeautifulSoup( article['ResultData']['News']['Content'] , 'html.parser').get_text().replace("(*#*)", "")] 
    keyword =  article['ResultData']['News'].get('Keywords', '').split(',')
    payload = json.dumps({
      "title": title,
      "content": content,
      "editor": keyword
    })
    headers = {
      'Content-Type': 'application/json'
    }
    url = "https://ckiptagger.cna.com.tw:8881/cut" #/origin #/captionCut #/forSpeech/
    response = requests.request("POST", url, headers=headers, data=payload)
    entity = json.loads(response.text) 

    entity_list = entity['ResultData']
    return entity_list # // output: list

# name convert
def name_convert_prompt(article_time, entity_list):
    # print("entity_list:\n",entity_list)
    name_prompt = """
    你 大語言模型 語料 中央社新聞報導
    任務 找出文章中的人物名稱 英文原文 和職稱

    ## 以下是偽代碼
    text = article_time
    person_list = entity_list['PERSON']

    1.根據person_list 比對text裡出現的人名，再判斷該人物的職稱或身分
        name_list = []
        for 人名 in person_list:
            if 人名 not in ['中央社記者', '編輯']:  # 排除特定角色
            
                職稱或身分 = 從原文中提取職稱或身分
                注意！一個人有多個職稱或身分 寫最完整的職稱或身分 或是合併相似的職稱
                注意！只列出person_list裡面出現的人名

                if 有text裡的人名有英文原文:
                    name_list.append(f"{中文人名}|{英文原文}|{職稱或身分}")
                else:
                    name_list.append(f"{中文人名}|{職稱或身分}")
        return name_list
        
    
    2. name = name_list.tolist()
    
    3. print(name)
        格式 ["{中文人名}|{職稱或身分}", "{中文人名}|{英文原文}|{職稱或身分}", ..."]
        注意！不需要標註 1. 2. 3. 等
"""

    convert_prompt = """
    你 大語言模型 語料 中央社新聞報導
    任務 根據name_list 置換text中的人名

    ## 以下是偽代碼
    text = article_time

    def replace_name(text, name_list):
        for 人名 in text:
            if 人名 in name_list:
                if 人名是譯名:
                    人名 = 人名.replace(人名, f"{中文人名}（{英文原文}，{職稱或身分}）")
                else:
                    人名 = 人名.replace(人名, f"{中文人名}（{職稱或身分}）")

        return text
    
    article_name = replace_name(text, name_list)

    注意！絕對不可以更動text的原文

    最終印出article_name
 """
    person_list = entity_list['entity']['PERSON']
    # 獲取 name_list 字符串
    name_list_str = generate_personName(name_prompt, article_time, person_list)
    
    # 將字符串轉換為列表
    try:
        # 移除可能的空白字符和換行符，並替換單引號為雙引號
        name_list_str = name_list_str.strip()
        if name_list_str.startswith('[') and name_list_str.endswith(']'):
            name_list = json.loads(name_list_str.replace("'", '"'))
        else:
            # 如果不是列表格式，可能是逐行輸出的格式
            name_list = [line.strip() for line in name_list_str.split('\n') if line.strip()]
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse name_list as JSON: {e}")
        # 創建一個備用的空列表
        name_list = []
    
    # print(">>>name_list:\n", name_list)
    
    article_name = generate_articleC(convert_prompt, article_time, json.dumps(name_list))
    # print(">>>人名置換後文章:\n", article_name)

    return article_name, name_list  # 現在返回的 name_list 是列表格式


# 清理文章內容
def clean_text(res, entity_list):

    strip_info = r'[（(][\w／\/:： ]*[）)]\d{0,8}$'
    story_dt = story_dt_cnt(res)
    news_content = res['ResultData']['News']['Content']
    # print("news content \n", news_content)
    paragraphs = re.sub(strip_info, '', BeautifulSoup(news_content.replace('(*#*)', ''), 'html.parser').text)
    article_time = date_noun_converter(story_dt, paragraphs)
    
    return  name_convert_prompt(article_time, entity_list)


# 保存文章數據
def get_article_data(pid):
    article_data = getArticle(pid)
    news = article_data['ResultData']['News']
    return {
        'pid': news['Id'],
        'published_dt': article_data['ResultData']['MetaData']['DateCreated'],
        'h1': news['Title'],
        'article': clean_text(article_data)[0],
        'extender': extender_counter(article_data),
        'category': news['TypeName'],
        'name_list': clean_text(article_data)[1] # 有需要加進去到這個嗎？
    }

# # 計算加權分數
# def calculate_weighted_scores(df):
#     df['ex_score'] = df['extender'] * 1.5
#     date_counts = df['published_dt'].value_counts().sort_values(ascending=False)
#     date_weights = {date: len(date_counts) - i for i, date in enumerate(date_counts.index)}
#     df['weighted_score'] = df.apply(lambda row: row['ex_score'] + date_weights[row['published_dt']], axis=1)
#     df = df.sort_values('weighted_score', ascending=False)
#     df_filtered = df[df['weighted_score'] > df['weighted_score'].max() / 2]
#     return df_filtered


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
def generate_date_convert(prompt, text, story_dt):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"請根據 story_dt: {story_dt} 轉換以下文本中的日期表述：\n\n{text}"}
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


# 生成人物辨識
def generate_personName(prompt,content, person_list):
    person_list_str = json.dumps(person_list, ensure_ascii=False)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content + person_list_str}
    ]
    return get_completion(messages)

def generate_articleC(prompt,content,name_res):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content + name_res}
    ]
    return get_completion(messages)

    
def count_num(tno):
    title_data = getTopic(tno)
    num = len(title_data['ResultData']['Items'])
    title_df = get_title_data(title_data)
    return num, title_df


