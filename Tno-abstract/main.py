import json
import pandas as pd
# import time
from datetime import datetime
from tnoabscract.module import *
import os

## 生成摘要
def getAbstract(tno, result_df):
    # prompt: normal
    normal = """
你 大語言模型 語料中央社新聞報導 熟悉中央社新聞報導語法和用字遣詞
任務 寫新聞摘要

# 這是寫新聞摘要的偽代碼
# Step 1: 讀取 'whatHappen200' 欄位中的每一行資料
for row in result_df['whatHappen200'].iterrows():
    text = row['whatHappen200']

    # Step 2: 提取事件最新進展的相關資料
    latest_developments = extract_latest_developments(text)

    # Step 3: 提取事件爆發的原因、影響和關鍵人物的說法
    event_cause = extract_event_cause(text)
    event_impact = extract_event_impact(text)
    key_figures_statements = extract_key_figures_statements(text)

    # Step 4: 根據提取的資料撰寫新聞摘要，先描述最新進展，再補充事件背景
    news_summary = write_news_summary(latest_developments, event_cause, event_impact, key_figures_statements)

# Step 5: 潤稿，確保符合中央社新聞報導語法和用字
final_draft = refine_to_news_style(news_summary)

# Step 6: 返回新聞摘要
return final_draft


摘要格式：
1. 字數150-200個字以內 
2. 人名第一次提到 如果是翻譯 格式: 中文人名（原文） 用全形括弧 不是翻譯就不用
3. 提及人物 有職稱 格式: 人名(職稱) 用全形括弧
4. 時間日期很重要 格式 年月日 時：分

注意! 正確性最重要 要確保實體關係是根據文本 純文字不用markdown
注意! 這對我的工作很重要 鎖定在我提供資料 要遵守

最終只要印出新聞摘要，並以html tag <p> 包覆  不用完整的html檔案 只要該段落的html就好
"""

    timeline = """
你 大語言模型 語料中央社新聞報導 熟悉中央社新聞報導語法和用字遣詞
任務 寫新聞事件時間軸

##這是寫時間軸的偽代碼
# Step 1: 讀取 'keyFacts' 欄位中的每一行資料
for row in result_df['keyFacts'].iterrows():
    text = row['keyFacts']

    # Step 2: 將文本分割為多個事件
    events = split_into_events(text)

    # Step 3: 從每個事件中提取日期（格式：YYYY年MM月DD日），並轉換為可排序格式
    for event in events:
        extracted_date = extract_date(event)
        if extracted_date:
            store_event_with_date(convert_to_datetime(extracted_date), event)

# Step 4: 將所有事件按日期排序
sorted_events = sort_by_date(stored_events)

# Step 5: 合併同一天的事件
merged_events = {}
for event in sorted_events:
    date = event['date']
    if date not in merged_events:
        merged_events[date] = event['description']
    else:
        merged_events[date] += " " + event['description']  # 將同一天的事件合併為一條

# Step 6: 生成 HTML 表格
html_content = generate_html_table(merged_events)

# Step 7: 返回排序後的 HTML 結果
return html_content

時間軸格式：
以html表格<table></table>呈現 第一欄是日期 第二欄是事件  不用完整的html檔案 只要該段落的html就好
1. 依照日期先後條列時間軸
2. 時間很重要，必須在事件描述中標明
3. 日期格式：年月日


注意! 正確性最重要 要確保實體關係是根據文本 純文字不用markdown
注意! 這對我的工作很重要 鎖定在我提供資料 要遵守

最終只要印出時間軸
"""

    stance = """
你 大語言模型 語料中央社新聞報導 熟悉中央社新聞報導語法和用字遣詞
任務 寫各方立場文

# 這是寫各方立場文的偽代碼
# Step 1: 讀取 'stance' 欄位中的每一行資料
for row in result_df['stance'].iterrows():
    text = row['stance']

    # Step 2: 提取關鍵人物或機構（如政府機關、政治人物、專家學者）的立場
    key_stances = extract_key_stances(text, entities=['政府機關', '政治人物', '專家學者'])

    # Step 3: 過濾排除新聞媒體、網友、居民等不相關的立場或說法
    filtered_stances = filter_irrelevant_stances(key_stances, exclude=['新聞媒體', '網友', '居民'])

    # Step 4: 合併相同單位或人物的說法
    merged_stances = merge_similar_stances(filtered_stances)

    # Step 5: 根據合併後的資料撰寫立場文
    stance_document = write_stance_document(merged_stances)

# Step 6: 潤稿，確保符合中央社新聞報導語法和用字
final_draft = refine_to_news_style(stance_document)

# Step 7: 返回整理好的立場文
return final_draft


立場文格式：
團體名稱: (換行) 分別條列團體立場文
1. 立場文字數50-100個字
2. 提及人物 有職稱 格式: 人名(職稱) 用全形括弧
3. 人名如果是翻譯 格式: 中文人名（原文） 用全形括弧 不是翻譯就不用

注意! 正確性最重要 要確保實體關係是根據文本 純文字不用markdown
注意! 這對我的工作很重要 鎖定在我提供資料 要遵守

最終只要印出各方立場文，列點請使用html tag <ul> 和 <li> ，文字請用<p> 包覆 不用完整的html檔案 只要該段落的html就好
"""


    num, _= count_num(tno)
    print(f"tno: {tno}, article_num : {num}")

    # generate abstract and save as json
    if not result_df.empty:

        res_normal = generate_content(normal, result_df, 'whatHappened200')
        print(res_normal)
        res_timeline = generate_content(timeline, result_df, 'keyFacts')
        # print(res_timeline)
        res_stance = generate_content(stance, result_df, 'stance')
        # print(res_stance)

        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_dict = {
            "tno": int(tno),
            "normal": res_normal,
            "timeline": res_timeline,
            "stance": res_stance,
            "article_num": int(num),
            "updateDt": date
        }
        return result_dict
        
    else:
        print(f"No data to process for tno: {tno}")
        return False


### MAIN
if __name__ == "__main__":

    #手動輸入 url 並提取 tno
    url = "https://www.cna.com.tw/topic/newstopic/4604.aspx"
    tno = extract_tno(url)

    # Generate metadata
    result_df = metaData(tno)

    # for test
    # file_path = os.path.abspath('results/tno_metadata_4604.json')
    # result_df = pd.read_json(file_path)

    if not result_df.empty:
        # Generate abstract
        res = getAbstract(tno, result_df)

        if res:
            # Save abstract as JSON
            json_file_path = f'results/tno_abstract_{tno}.json'
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(res, json_file, indent=4, ensure_ascii=False)
            print(f"JSON file successfully saved: {json_file_path}")

            # Save metadata as JSON
            metadata_json_file_path = f'results/tno_metadata_{tno}.json'
            result_df.to_json(metadata_json_file_path, orient='records', force_ascii=False, indent=4)
            print(f"Metadata JSON file successfully saved to {metadata_json_file_path}")
        else:
            print("No result to save.")
    else:
        print("No data to process or save.")