import pandas as pd
from datetime import datetime
import time
import re
from module import *
# from aiFunc import *
# from articleFunc import notify
from sheetFunc import *
import os
import traceback
import argparse
from es_SearchLib import *
# from metadata import *


# 從elastic search拿資料存成tempfile
def get_elastic(done, ago=30):
    # 設定時間計算
    pattern_today = f"{datetime.today().strftime('%Y-%m-%d')}"
    thirty_mins_ago = datetime.now() - timedelta(minutes=ago)
    # 用特定時間拿elastic search的資料
    from_elastic = es_search_certain_date(es, index="lab_mainsite_search", date_column_name="dt", date=pattern_today)
    # 把符合時間計算的pids存成一個list
    pids = [i['_source']['pid'] for i in from_elastic if datetime.strptime(i['_source']['dt'], "%Y/%m/%d %H:%M") < thirty_mins_ago and i['_source']['pid'] not in done]
    
    data = []
    for pid in pids:
        try:
            # 用pid當作index過濾資料
            es_res_list = [i['_source'] for i in from_elastic if i['_source']['pid'] == pid]
            if es_res_list:
                for es_res in es_res_list: # es_res is a dict

                    d = {
                        'pid' : pid, 
                        'dt': es_res.get('dt', ''),
                        'h1': es_res.get('h1', ''),
                        'article': es_res.get('article', ''),
                        'text': es_res.get('text', ''), 
                        'embeddings' : es_res.get('embeddings', []),
                        'focus': es_res.get('foucus', 0),
                        'extender': es_res.get('extender', 0),
                        'attachments': es_res.get('attachments', 0),
                        'category': es_res.get('category', '')
                    }

                    data.append(d)
                    print(f'{pid} had add into data.')
                    done.append(pid)
            
        except Exception as e:
            print(f"Error processing pid {pid}: {e}")
            continue

    if data:
        tempfile = f"embeded/df-temp-{datetime.today():%Y%m%d}.csv"
        df = pd.DataFrame(data)
        print(f'{datetime.today():%Y%m%d} data already saved.')
        print(f'df include {len(df)} articles.')
        df.to_csv(tempfile, sep='\t', encoding='utf-8', index=False)
        return df


# summarize prompt
def summarizeGPT1(df):  # df已經是有metadata的資料了
    prompt01 = """
    你是大語言模型，熟悉中央社新聞報導的語法和用字遣詞
    任務 綜合數篇文章寫100個字以內的新聞電子報摘要

    依照以下步驟執行：
    1. 讀取df裡，每一個pid文章的 whatHappen200、keyFacts和stance
    2. 判斷新聞的核心內容 包含但不限於：事件最新進展、爆發原因、關鍵事件日期、相關人物說法等
    3. 改寫成50-100個字新聞摘要。以新聞電子報格式書寫，先寫事件最新進展，再補充爆發原因、影響、關鍵人物說法等。

    電子報摘要格式：
    1. 寫作必須符合中央社新聞報導的語法和用字遣詞。寫完要潤稿
    2. 人名第一次提到 如果是翻譯 格式: 中文人名（原文） 用全形括弧 不是翻譯就不用
    3. 提及人物 有職稱 格式: 人名(職稱) 用全形括弧
    4. 時間日期很重要 格式 年月日 時：分

    注意! 正確性最重要 要確保實體關係是根據文本 純文字不用markdow
    注意! 這對我的工作很重要 鎖定在我提供資料 要遵守
    
    最終印出電子報摘要結果
    注意！移除開頭的"電子報摘要結果"等說明，直接print摘要結果
    """
    summary1 = generate_morning(prompt01, df)
    print(f'summary 1:, \n{summary1}')
    

    # 返回兩個摘要版本的列表
    return summary1


def summarizeGPT2(df):
    prompt02 = """
    你 大語言模型 語料中央社新聞報導 熟悉中央社新聞報導語法和用字遣詞
    任務 寫新聞摘要

    # 這是偽代碼
    text = df['whatHappen200'].itterows()
    利用text的資料改寫成一則新聞摘要：先寫事件最新進展，再補充事件爆發原因、影響、關鍵人物說法等。
    寫完必須潤稿，書寫符合中央社新聞報導語法和用字
    摘要格式：
    1. 字數50-100個字以內 
    2. 人名第一次提到 如果是翻譯 格式: 中文人名（原文） 用全形括弧 不是翻譯就不用
    3. 提及人物要寫：職稱人名
    4. 時間日期很重要 格式 年月日 時：分

    注意! 正確性最重要 要確保實體關係是根據文本 純文字不用markdow
    注意! 這對我的工作很重要 鎖定在我提供資料 要遵守

    注意！移除開頭的"電子報摘要結果"等說明
    最終直接print摘要結果
    """

    summary2 = generate_morning(prompt02, df)
    print(f'summary 2:, \n{summary2}')

    return summary2


# Summarize
def summarize(df, group, summary_funcs, authFilePath, fileId, sheet_name):
    summaries = []
    for i, c in enumerate(group):
        try:
            print(f'正在處理群組 {i+1} / {len(group)}')
            data = df.loc[df['cluster_first_stage'] == c].sort_values(by='sort_cri', ascending=False).reset_index()
            n = len(data) if len(data) < 4 else int(len(data) * 0.5)

            h1 = data['h1'].tolist()[:n]

            # 建立包含 whatHappen, keyFacts, stance 的 DataFrame
            paragraphs = pd.DataFrame({
                'whatHappen': data['whatHappen200'][:n],  
                'keyFacts': data['keyFacts'][:n],  
                'stance': data['stance'][:n]        
            })

            # 調用生成摘要的函數，傳入 DataFrame
            summary1 = summary_funcs[0](paragraphs)
            time.sleep(2)
            summary2 = summary_funcs[1](paragraphs)
            time.sleep(2)

            summ_data = [str(h1), summary1, summary2]
            update_df(authFilePath, fileId, sheet_name, 'append', '', [summ_data], header=False)  # 更新摘要工作表
            summaries.append(summ_data)

        except:
            trace = traceback.format_exc()
            ind = re.search('line \d*\,', trace).span()[0]
            # notify(f"AI寫摘要失敗，錯誤訊息：{trace[ind:]}", token)
            print(f"AI寫摘要失敗，錯誤訊息：{trace[ind:]}")

    # 處理未被歸類的零散文章摘要
    single_num = 25 - len(group) if len(group) < 25 else 5
    ungrouped_articles = df.loc[df['cluster_first_stage'] < 0].sort_values(by='sort_cri', ascending=False).reset_index()[0:single_num]

    for i, s in enumerate(ungrouped_articles.index[0:single_num]):
        try:
            print(f'正在處理零散文章 {i+1} / {single_num}')
            data = [ungrouped_articles.loc[s, 'article']]
            h1 = ungrouped_articles.loc[s, 'h1']

            # 建立單篇文章的 DataFrame
            paragraphs = pd.DataFrame({
                'whatHappen': [ungrouped_articles.loc[s, 'whatHappen200']], 
                'keyFacts': [ungrouped_articles.loc[s, 'keyFacts']], 
                'stance': [ungrouped_articles.loc[s, 'stance']]        
            })

            # 調用生成摘要的函數
            summary1 = summary_funcs[0](paragraphs)
            time.sleep(2)
            summary2 = summary_funcs[1](paragraphs)
            time.sleep(2)

            summ_data = [str(h1), summary1, summary2]
            update_df(authFilePath, fileId, sheet_name, 'append', '', [summ_data], header=False)  # 更新摘要工作表
            summaries.append(summ_data)

        except:        
            trace = traceback.format_exc()
            ind = re.search('line \d*\,', trace).span()[0]
            # notify(f"AI寫摘要失敗，錯誤訊息：{trace[ind:]}", token)
            print(f"AI寫摘要失敗，錯誤訊息：{trace[ind:]}")

    return summaries



## Main
if __name__ == '__main__':
    authFilePath = os.getenv("GOOGLE_CREDENTIALS_JSON")
    fileId = os.getenv("File_ID")
    # web_token = os.getenv("notifyWeb")
    # token = os.getenv("line_notify_token")

    parser = argparse.ArgumentParser(description='Process some Var')
    parser.add_argument("-t", type=str, help="任務", default='ohayo')
    #parser.add_argument("-u", type=bool, help="摘要上傳", default=True)
    # parser.add_argument("-n", type=bool, help="通知編輯", default=True)
    args = parser.parse_args()


    if args.t == 'ohayo':
        sheet_name = 'summarized'
        article_sheet = 'articles'
        keywords_sheet = 'keywords today' 
        tempfile = f'embeded/df-temp-{datetime.today():%Y%m%d}.csv'
        temp_df = get_elastic(done=[], ago=30) #elastic search data
        df = metaData(temp_df)  # add metadata into temp_df
        # web_notify_msg = f"請來看AI【志明寫早報】幫忙的摘要~ https://tinyurl.com/ymenj8qh" 
        func = [summarizeGPT1, summarizeGPT2] 
        focus = False
        # gTrends = -3

    df, group = df_tagging(df, focus, keywords=keywords_sheet)# 暫時移除gTrends

    #移除temp_df的embeddings欄位
    temp_to_save_df = temp_df[['pid', 'dt', 'h1', 'article', 'text', 'focus', 'extender', 'attachments', 'category']]

    # save result to spreadsheet
    sheetUpdate(authFilePath, fileId, article_sheet, df=temp_to_save_df) 
    sheet_clear_values(authFilePath, fileId, sheet_name, "A2:Z") 
    summaries = summarize(df, group, func, authFilePath, fileId, sheet_name) 

    # save summarized result into csv
    df1 = pd.DataFrame(summaries, columns=['title', 'ver1', 'ver2'])
    save_file = f'summarized/summarized-{datetime.today():%Y%m%d}.csv'
    df1.to_csv(save_file, sep='\t', encoding='utf-8', index=False)


    # if args.u:
    #     sheetUpdate(authFilePath, fileId, sheet_name, df1)
    # if args.n:
    #     notify(web_notify_msg, web_token)


