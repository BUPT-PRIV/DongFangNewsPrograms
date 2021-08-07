import json

with open("result/1.json", encoding='utf-8') as f:
    data_all = json.load(f)
    for key in data_all:
        news = data_all[key]
        news["EndPosition"] = int(news["EndPosition"])*10
        data_all[key] = news
    res = json.dumps(data_all, indent=4, ensure_ascii=False)
    f_res = open("result/1.json", "w", encoding='utf8')
    f_res.write(res)
