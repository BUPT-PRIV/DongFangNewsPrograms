import json
import os
import sys
from typing import Text
import time
import argparse

file_path = dataset_dir = os.path.dirname(
    os.path.abspath("__file__"))

sys.path.append(os.path.join(file_path,'silence_seg/'))
os.chdir(os.path.join(file_path,'silence_seg/'))
from weblfasr_python3_demo import *

sys.path.append(os.path.join(file_path,'berts_pytorch_zh/'))
os.chdir(os.path.join(file_path,'berts_pytorch_zh/'))
from concat_text import *


sys.path.append(os.path.join(file_path,'Bert-10-class/'))
os.chdir(os.path.join(file_path,'Bert-10-class/'))
from infone10 import *

sys.path.append(os.path.join(file_path,'Bert-3-class/'))
os.chdir(os.path.join(file_path,'Bert-3-class/'))
from infone3 import *

sys.path.append(os.path.join(file_path,'Bert-3000-class/'))
os.chdir(os.path.join(file_path,'Bert-3000-class/'))
from infone3000 import *

sys.path.append(os.path.join(file_path,'GPT2-Summary/'))
os.chdir(os.path.join(file_path,'GPT2-Summary/'))
from tosum import sumone

def news_seg(video_files):
    start=time.time()
    dir_list=os.listdir(video_files)
    if not dir_list:
        return
    remove_file=os.path.join(file_path,"berts_pytorch_zh/result")
    for i in os.listdir(remove_file):
        path_file = os.path.join(remove_file,i) 
        if os.path.isfile(path_file):
            os.remove(path_file)

    for cur_video in dir_list:
        appid="a0dfb902"
        secret_key="481bf739434724803910ef45f35384bb"

        
        video_file=os.path.join(video_files,cur_video)
        print(video_file)
        os.chdir(os.path.join(file_path,'silence_seg/'))
        print(os.getcwd())
        
        audio_file = "data/audio/"+os.path.splitext(cur_video)[-2]+".mp3"
        json_path="result/"+os.path.splitext(cur_video)[-2]+".json"

     

        run(video_file,audio_file,appid,secret_key,json_path)

        text=Text()
        text.filename=os.path.join(os.getcwd(),json_path)
        text.video_file=os.path.join(os.getcwd(),video_file)

        os.chdir(os.path.join(file_path,'berts_pytorch_zh/'))
        print(os.getcwd())

       
       
        single_result_file = os.path.join(os.getcwd(),"result/"+os.path.splitext(cur_video)[-2]+".json")
        text.save_path=single_result_file
        text.get_concat_condition()
    
    result_path=os.path.join(os.getcwd(),"result")
    result_file = os.path.join(file_path,"middle.json")
    data_list_total=[]
    with open(result_file,"w",encoding="utf-8") as f:
        for file in os.listdir(result_path):

            with open(os.path.join(result_path,file),"r",encoding="utf-8") as ff:
               
                data_all = json.load(ff)
                # 读取json文件
                #将字典的值转为列表
                data_list = data_all["answer"]
                data_list_total.extend(data_list)
                ff.close()
        
        for index in range(len(data_list_total)):
            data_list_total[index]["news_id"]=index+1
        
        data_all = {}
        data_all["answer"] = data_list_total
        
        res = json.dumps(data_all, indent=4, ensure_ascii=False)
        f.write(res)
        f.close()

    end=time.time()
    print("news_seg time:",str(end-start))

def clsandtip():
    os.chdir(file_path)
    with open(os.path.join(file_path,"middle.json"),'r') as f:
        inputs = json.load(f)
    inputs = inputs['answer']
    for j,input in enumerate(inputs):
        enhance = findsametip(input['content'])
        print('start:'+str(j))
        os.chdir(os.path.join(file_path,'GPT2-Summary/'))
        content = input['content']
        tsum = sumone(content)
        print(tsum)
        os.chdir(os.path.join(file_path,'Bert-3-class/'))
        cls3 = testone3(tsum)
        os.chdir(os.path.join(file_path,'Bert-10-class/'))
        cls10 = testone10(tsum)
        input['category'] = cls3+cls10
        os.chdir(os.path.join(file_path,'Bert-3000-class/'))
        cls3000 = testone3000(tsum)
        cls3000 = getfinal(cls3000,enhance)
        input['tags'] = cls3000
        print(cls3000)
        del input['content']
    os.chdir(file_path)
    with open('result.json','w') as f:
        json.dump(inputs,f,ensure_ascii=False)

def getfinal(cls,enhance):
    enh4 = []
    enhmax = []
    maxlen = 1
    if len(enhance)==0:
        return cls
    for each in enhance:
        if len(each)>maxlen and each not in cls:
            maxlen = len(each)
    if maxlen==4:
        if len(cls[3])<4:
            for each in enhance:
                if len(each)>=4 and each not in cls:
                    enh4.append(each)
            if len(enh4) == 1:
                cls[3]=enh4[0]
    elif maxlen >4:
        if len(cls[3])<=4:
            for each in enhance:
                if len(each)==maxlen and each not in cls:
                    enhmax.append(each)
            cls[3] = enhmax[0]
    return cls

def findsametip(txt):
        os.chdir(file_path)
        with open('class.txt','r') as f:
            tips = f.readlines()
        for i in range(len(tips)):
            tips[i] = tips[i].strip('\n')
            if tips[i] == '':
                tips.remove('')
        answer = []
        for tip in tips:
            if tip in txt:
                answer.append(tip)
        return answer
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--path', type=str, default='/home/zhouyang/Train_and_test/data_test', help='输入绝对路径')
args = parser.parse_args()

if __name__ == '__main__':
    path = args.path
    news_seg(path)
    clsandtip()
        