import os
from dotenv import load_dotenv
import requests
import json
from datetime import datetime
import openai
import functions_framework
import flask
from google.cloud import storage




# open ai setting
load_dotenv()
openai_api_key = os.getenv("Chiming_KEY")

def get_completion(messages, model="gpt-4", temperature=0):
    payload = { "model": model, "temperature": temperature, "messages": messages}
    headers = { "Authorization": f'Bearer {openai_api_key}', "Content-Type": "application/json" }
    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, data=json.dumps(payload))
    obj = json.loads(response.text)
    
    # 添加錯誤處理
    if 'error' in obj:
        print(f"Error: {obj['error']['message']}")
        return None
    
    # 返回生成的文本
    return obj['choices'][0]['message']['content']

# initialize
# for generate metadata
def generate_metadata(summarise, article, entity_list, name_list):
    formatted_content = (
        f"{article}\n"
        f"實體列表：{json.dumps(entity_list, ensure_ascii=False)}\n"
        f"名稱列表：{', '.join(name_list)}"
    )
    messages = [
        {"role": "system", "content": summarise},
        {"role": "user", "content": formatted_content}
    ]
    return get_completion(messages)


def meta_prompt(article, entity_list, name_list):
    print("entity_list:\n",entity_list)
    print("name_list:\n",name_list)
    # prompt: what happen?
    whatHappen50 = """
    你 大語言模型 語料中央社新聞報導 熟悉中央社新聞報導語法和用字遣詞
    任務 閱讀文章後寫50字新聞摘要

    ## 以下是偽代碼
    text = article
    key_entity = entity_list['keywords']

    1. 從text 判斷新聞核心事件或議題 包含但不限 事件名稱、相關人物、關鍵事件、日期時間
        news_event = []
        for keyword in key_entity:
            if keyword in text:
                news_event.append(keyword + " 相關事件摘要")
            
    2. 利用news_event 整理新聞摘要
        格式：50個字以內、符合中央社新聞書寫
        人物書寫格式要符合 name_list

    3. 最終只要印出事件摘要結果

    注意！正確性最重要 要確保實體關係是根據文本 純文字不用markdow
    請注意！這對我的工作很重要 鎖定在我提供資料 要遵守
    """

    whatHappen200 = """
    你 大語言模型 語料中央社新聞報導 熟悉中央社新聞報導語法和用字遣詞
    任務 閱讀文章寫200字摘要

    ## 以下是偽代碼
    text = article
    key_entity = entity_list['keywords']

    1. 從text 判斷新聞核心事件或議題 包含但不限 事件名稱、相關人物、關鍵事件、日期時間
        news_event = []
        for keyword in key_entity:
            if keyword in text:
                news_event.append(keyword + " 相關事件摘要")
            
    2. 利用news_event 整理新聞摘要
        格式：150-200個字以內、符合中央社新聞書寫
        人物書寫格式要符合 name_list

    3. 最終只要印出事件摘要結果

    注意！正確性最重要 要確保實體關係是根據文本 純文字不用markdow
    請注意！這對我的工作很重要 鎖定在我提供資料 要遵守
    """

    # prompt: what are the key facts?
    keyFacts = """
    你 大語言模型 語料中央社新聞報導 熟悉中央社新聞報導語法和用字遣詞
    任務 閱讀文章分條列出關鍵事件(key_facts)

    ## 以下是偽代碼
    text = article
    key_entity = entity_list['keywords']

    1. 判斷關鍵事件 目的要凸顯事件發生經過，包含但不限：發生事件名稱 日期 地點 時間 
       key_facts = []
        for keyword in key_entity:
            if keyword in text:
                判斷跟keyword有關的事件時間、事件摘要 
                key_facts.append("日期" + " keyword" + " 事件摘要")

    2. 分條列出key_facts
        格式 日期：(換行) keyword + " 事件摘要" 
        日期格式 = datetime.strptime(date_str, "%Y-%m-%d") 如果沒有日期就空著日期欄位
        人物書寫格式要符合 name_list
    3. 最終只要印出key_facts

    注意！正確性最重要 要確保實體關係是根據文本 純文字不用markdow
    請注意！這對我的工作很重要 鎖定在我提供資料 要遵守
    """

    # prompt: what key person say?
    stance = """
    你 大語言模型 語料中央社新聞報導 熟悉中央社新聞報導語法和用字遣詞
    任務 閱讀文章寫各方立場文

    ## 以下是偽代碼
    text = article
    key_entity = entity_list['keywords']

    1. 判斷各方立場文 目的在說明相關人或機構的立場、說法，包含但不限：相關人或機構名稱 說法、看法、立場
        stance_info = []
        for e in key_entity:
            if e in text:
                判斷跟e有關的立場文
                stance_info.append("團體/人名" + " 立場")

    2. 分段列出各方stance_info 100個字以內
        格式: 團體/人名: (換行) 立場文
        人物書寫格式要符合 name_list

    3. 最終只要印出各方立場文

    注意！正確性最重要 要確保實體關係是根據文本 純文字不用markdow
    請注意！這對我的工作很重要 鎖定在我提供資料 要遵守
    """
    key_entity = entity_list['keywords']

    what_happen_50 = generate_metadata(whatHappen50, article, key_entity, name_list)
    print(">>> what_happen_50:\n",what_happen_50)
    what_happen_200 = generate_metadata(whatHappen200, article, key_entity, name_list)
    key_facts = generate_metadata(keyFacts, article, key_entity, name_list)
    stance_info = generate_metadata(stance, article, key_entity, name_list)
    

    result = {
        "article_converted": article,
        "whatHappen50": what_happen_50,
        "whatHappen200": what_happen_200,
        "keyFacts": key_facts,
        "stance": stance_info
    }
    
    return json.dumps(result, ensure_ascii=False)
    
# save to cloud storage
storage_client = storage.Client()
bucket_name = 'flask-project'
tempfile = f'metadata_{datetime.now().strftime("%Y%m%d%H%M%S")}'
file_path = f'generate-metadata/meta_result/metadata_{tempfile}.json'

def save_json_to_cloud_storage(bucket_name, file_path, data):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    blob.upload_from_string(
        data=json.dumps(data, ensure_ascii=False, indent=4),
        content_type='application/json'
    )
    print(f"File {file_path} saved to {bucket_name}.")


## Main
@functions_framework.http
def metadata(request: flask.Request):
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',  # Allow your function to be called from any domain
            'Access-Control-Allow-Methods': 'POST',  # POST or any method type you need to call
            'Access-Control-Allow-Headers': 'Content-Type', 
            'Access-Control-Max-Age': '3600',
        }
        return ('', 204, headers)

    # Set CORS headers for main requests
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
    }

    try:
        data = request.get_json()
        
        # check if content, entity_list, and name_list is in data(json)
        if not all(key in data for key in ['content', 'entity_list', 'name_list']):
            return flask.jsonify({"error": "Missing required fields: content, entity_list, or name_list"}), 400

        article = "".join(data['content'])
        entity_list = data['entity_list'] 
        name_list = data['name_list']

        result = meta_prompt(article, entity_list, name_list)
        if not result:
            return flask.jsonify({"error": "Failed to generate metadata"}), 500

        try:
            save_json_to_cloud_storage(bucket_name, file_path, result)
        except Exception as e:
            print(f"Storage error: {str(e)}")
            
        return flask.jsonify(json.loads(result)), 200

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return flask.jsonify({"error": f"Internal server error: {str(e)}"}), 500
