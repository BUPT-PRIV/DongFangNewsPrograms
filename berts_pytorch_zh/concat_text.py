import json
import os
from os import EX_DATAERR
from typing import Tuple

from sqlalchemy.sql.expression import true
from network.bert_vec import *
import numpy as np
from scipy.linalg import norm
from fun_all import *
import cv2


class Text:
    def __init__(self):
        self.start = "Start_frameID"
        self.end = "end_frameID"
        self.content = "content"
        self.title = False
        self.video_file=""
        self.filename=""
        self.save_path=""

    # #根据图像调整开始结束
    # def adjust_result(self,data_list):
        

    # 合并前后短句
    def concat_text(self, pre_index, data_list, index):
        news = data_list[index]
        data_list[pre_index][self.end] =news[self.end]
        data_list[pre_index][self.content] = data_list[pre_index][self.content] + news[self.content]
        data_list.pop(index)  # 删除上一条

    #按照stride合并
    def concat_text_stride(self, stride, data_list, index):
        if index + stride >= len(data_list):
            data_list[index][self.end] = data_list[len(data_list) -
                                                   1][self.end]
        else:
            data_list[index][self.end] = data_list[index + stride][self.end]
        for i in range(stride):
            if index + i < len(data_list):
                data_list[index][self.content] += data_list[index +
                                                            i][self.content]
                data_list.pop(index + i)  #删除合并的list内容

    #保存json文件
    def save_json(self, data_all):
        res = json.dumps(data_all, indent=4, ensure_ascii=False)
        f_res = open(self.save_path, "w", encoding='utf8')
        f_res.write(res)
        return f_res

    #获得所有本文向量
    def vec_calculate(self, data_list):
        data_vec = []
        # 模型初始化
        model = BertTextNet()  # 选择一个文本向量化模型
        seq2vec = BertSeqVec(model)  # 将模型实例给向量化对象。

        for index in range(len(data_list)):
            s = data_list[index][self.content]
            #语料向量化
            vec = seq2vec.seq2vec(s)
            data_vec.append(vec)
        return data_vec

    #计算向量相似度
    def text_similarity_pre(self, s1, s2):
        # 模型初始化
        model = BertTextNet()  # 选择一个文本向量化模型
        seq2vec = BertSeqVec(model)  # 将模型实例给向量化对象

        if len(s1)>500:
            s1=s1[0:500]
        if len(s2)>500:
            s2=s2[-500:]        

        v1 = seq2vec.seq2vec(s1)
        v2 = seq2vec.seq2vec(s2)

        score = text_similarity(v1, v2)
        print(score)

        return score

    #根据步长计算文本向量相似度
    def stride_similarity_cal(self, stride, data_list, t):
        index = 0
        #获得所有文本数据的向量
        data_vec = self.vec_calculate(data_list)
        s = stride
        while index < len(data_list):
            content = data_list[index][self.content]
            if content[0] == '嗯' and content.count('嗯') >= 2 or content == '嗯':
                index += 1
                continue
            if s == 0:
                index += 1
                s = stride
            vec_cur = data_vec[index]
            if (index + s >= len(data_list)):
                vec_next = data_vec[len(data_list - 1)]
            vec_next = data_vec[index + s]
            score = text_similarity(vec_cur, vec_next)
            # 如果两本文向量的相似度大于等于阈值，则合并其中所有的文本
            if score >= t:
                self.concat_text_stride(s, data_list, index)
                index += 1
            # 否则将步长-1，继续判断
            else:
                s -= 1
        return data_list
    
    def list_process(self,data_list):
        cap=cv2.VideoCapture(self.video_file)
        fps=cap.get(cv2.CAP_PROP_FPS)
        for i in range(len(data_list)):
            data_list[i][self.start]=int(fps*int(data_list[i][self.start])/1000)
            data_list[i][self.end]=int(fps*int(data_list[i][self.end])/1000)
            data_list[i]["video_name"] = os.path.basename(self.video_file).split('.')[0]
            data_list[i]["news_id"] = i + 1


    #删除之前的数据的关键词判断
    def del_forward(self,cur):
        del_forward=False
        if "来看"in cur[self.content] and "详细" in cur[self.content] and ("报道" in cur[self.content] or "内容" in cur[self.content]):
            del_forward=True
            return del_forward
        if cur[self.content].endswith("向各位问好。"):
            del_forward=True
            return del_forward
        if cur[self.content].startswith("观众") and "欢迎收看" in cur[self.content]:
            del_forward=True
            return del_forward
        return del_forward

    #删除中间部分的初始词判别
    def del_middle_pre(self,cur):
        del_middle=False
        if "这里是" in cur[self.content] and "稍后" in cur[self.content]:
            del_middle=True
            return del_middle
        if "稍后" in cur[self.content] and "回来" in cur[self.content]:
            del_middle=True
            return del_middle
        if "请锁定" in cur[self.content] and "稍后" in cur[self.content]:
            del_middle=True
            return del_middle
        if "广告" in cur[self.content] and ("稍后" in cur[self.content] or "继续" in cur[self.content]) :
            if "广告回来" in cur[self.content]:
                del_middle=False
            else:
                del_middle=True
            return del_middle
        if cur[self.content].startswith("这里是正在") and "直播" in cur[self.content] and "将" in cur[self.content]:
            del_middle=True
            return del_middle
    
    #删除中间部分的结束词判别
    def del_middle_end(self,cur):
        del_middle=False
        if "欢迎回来" in cur[self.content]:
            del_middle=True
            return del_middle
        if "广告回来" in cur[self.content] or "进入夜线":
            del_middle=True
            return del_middle
        if cur[self.content].startswith("好") and "转回" in cur[self.content]:
            del_middle=True
            return del_middle
        if cur[self.content].startswith("来看今天的夜线关注"):
            del_middle=True
            return del_middle
        return del_middle
    
    #合并被“嗯”分割的同一新闻
    def concat_split_news(self,data_list):
        index=0
        start=end=0
        while index+1<len(data_list):
            if data_list[index][self.content]!="嗯" and data_list[index+1][self.content]=="嗯":
                start=index
            elif data_list[index][self.content]=="嗯" and data_list[index+1][self.content]!="嗯":
                end=index
            if start<end and start!=0 and end!=0:
                if tag_extract(data_list[start][self.content], data_list[end][self.content],15,2):
                    for i in range(start+1,end):
                        data_list.pop(start+1)
                    data_list[start][self.end]=data_list[start+1][self.end]
                    data_list[start][self.content]+=data_list[start+1][self.content]
                start=end=0
            index+=1
   
    #根据长短删除其他内容
    def del_other(self,data_list):
        index=0
        while index<len(data_list):
            if len(data_list[index][self.content])<80:
                data_list.pop(index)
            elif "幸运观众" in data_list[index][self.content] and "好礼" in data_list[index][self.content]:
                data_list.pop(index)
            else:
                index+=1

    #删除中间无关部分
    def del_middle_content(self,data_list):
        start=[]
        end=[]
        #确定删除区域
        for index in range(len(data_list)):
            content=data_list[index]
            if self.del_middle_pre(content):
                start.append(index)
                continue
            if len(start)==len(end)+1 and self.del_middle_end(content):
                end.append(index)
        #删除其他部分
        if len(start)!=0 and len(end)!=0 and len(end)==len(start):
            for i in range(len(start)):
                n=end[i]-start[i]
                for j in range(n):
                    data_list.pop(start[i])
                j=i+1
                for j in range(i+1,len(start)):
                    start[j]-=n
                    end[j]-=n
        
   
    #天气判断（结束新闻播报判断）
    def weather_judge(self,cur):
        is_end=False
        if "再见" in cur[self.content] and "感谢" in cur[self.content]:
            is_end=True
            return is_end
        if "以上" in cur[self.content] and "全部" in cur[self.content]:
            is_end=True
            return is_end
        if "就到这里" in cur[self.content] and "节目" in cur[self.content]:
            is_end=True
            return is_end  
        if cur[self.content].startswith("好") and "今天" in cur[self.content] and "聊到" in cur[self.content]: 
            is_end=True
            return is_end  
        return is_end

    #删除天气预报部分 
    def concat_weather(self,data_list,weather_begin):
        if weather_begin!=0 and weather_begin<len(data_list):
            # data_list[weather_begin][self.end]=data_list[-1][self.end]
            # n=len(data_list)-weather_begin-1
            # for i in range(n):
            #     data_list[weather_begin][self.content]+=data_list[weather_begin+1][self.content]
            #     data_list.pop(weather_begin+1)  # 删除
            n=len(data_list)-weather_begin
            for i in range(n):
                data_list.pop(weather_begin)  # 删除

        
    #合并词判别
    def word_concat(self, pre, cur):
        concat_s = False
        if cur[self.content].endswith("报道。") or cur[self.content].endswith("报道，") or pre[self.content].endswith( "据报道，" ):
            concat_s = True
            return concat_s
        if len(cur[self.content]) < 5:
            concat_s = True
            return concat_s
        if not pre[self.content].endswith('。'):
            concat_s = True
            return concat_s
        if len(cur[self.content]) < 7:
            concat_s = True
            return concat_s
        if cur[self.content][0] == "而" or cur[self.content][0]=="但" or cur[self.content].startswith("另外") or cur[self.content].startswith("所以"): 
            concat_s = True
            return concat_s
        if ord(cur[self.content][0]) in range(65,91) or ord(cur[self.content][0]) in range(97,123):
            concat_s=True
            return concat_s
        if cur[self.content].startswith("我也") or cur[self.content].startswith("嗯") or pre[self.content].endswith("啊"):
            concat_s=True
            return concat_s
        if cur[self.content].startswith("下一步") or cur[self.content].startswith("最后") or cur[self.content].startswith("呃"):
            concat_s=True
            return concat_s
        if cur[self.content].startswith("经") or cur[self.content].startswith("你") or cur[self.content].startswith("从") or cur[self.content].startswith(" "):
            concat_s=True
            return concat_s
        return concat_s
    
    #间隔时间判断
    def time_judge(self,pre,cur):
        cur_start = cur[self.start]
        pre_end = pre[self.end]
        dur_time = int(cur_start) - int(pre_end)
        time_long = False
    
        if dur_time >= 1300:
            time_long = True
        return time_long

    #不合并词判别
    def word_disconcat(self, pre, cur):
        cur_start = cur[self.start]
        pre_end = pre[self.end]
        dur_time = int(cur_start) - int(pre_end)
        disconcat_s = False
        if (cur[self.content][0] == '嗯' and cur[self.content].count(
                '嗯') >= 2 and len(cur[self.content])<=5) or cur[self.content] == '嗯':
            disconcat_s = True
            return disconcat_s
        if (pre[self.content][0] == '嗯' and pre[self.content].count(
                '嗯') >= 2 and len(pre[self.content])<=5) or pre[self.content] == '嗯':
            disconcat_s = True
            return disconcat_s
        if pre[self.content].endswith("报道。") or pre[self.content].endswith("报道，"):
            if  pre[self.content].endswith("据报道，") or pre[self.content].endswith("来看报道。") or pre[self.content].endswith("据美国全国广播公司NBC报道，") or pre[self.content].endswith(
                "来看记者的报道。")  or pre[self.content].endswith("特别报道。") or pre[self.content].endswith("发回的报道。"):
                disconcat_s=False
                return disconcat_s
            disconcat_s = True
            return disconcat_s
        if "谢谢" in pre[self.content] and cur[self.content].startswith("好"):
            disconcat_s = True
            return disconcat_s
        if cur[self.content].startswith("好") and ("来看" in cur[self.content] or "感谢收看" in cur[self.content]):
            disconcat_s = True
            return disconcat_s
        if cur[self.content].startswith("好") and ("广告" in cur[self.content] or "转回" in cur[self.content]):
            disconcat_s = True
            return disconcat_s
        if  cur[self.content].startswith("接下来"):
            if cur[self.content].startswith("接下来除了") or cur[self.content].startswith("接下来可能") or cur[self.content].startswith("接下来更"):
                disconcat_s=False
                return disconcat_s
            else:
                disconcat_s = True
                return disconcat_s
        if cur[self.content].startswith("好") and "最后" in cur[self.content]:
            disconcat_s = True
            return disconcat_s
        if cur[self.content].startswith("好") and ("以上" in cur[self.content] or "接下来" in cur[self.content]):
            disconcat_s = True
            return disconcat_s
        if "马上回来" in pre[self.content] or  "欢迎回来。" in pre[self.content]:
            disconcat_s = True
            return disconcat_s
        if "我们" in cur[self.content] and "转向" in cur[self.content]:
            disconcat_s = True
            return disconcat_s
        if "我们" in cur[self.content] and "转回" in cur[self.content]:
            disconcat_s = True
            return disconcat_s
        if cur[self.content].startswith("这里是") and ("直播" in cur[self.content] or "我是" in cur[self.content] or "来看" in cur[self.content]):
            disconcat_s = True
            return disconcat_s
        if "就介绍到这里" in pre[self.content]:
            disconcat_s = True
            return disconcat_s
        if cur[self.content].startswith("我们") and "其他" in cur[self.content]:
            disconcat_s = True
            return disconcat_s
        if "请锁定" in cur[self.content] and "稍后" in cur[self.content]:
            disconcat_s = True
            return disconcat_s
        if "以上" in cur[self.content] and "稍后" in cur[self.content]:
            disconcat_s = True
            return disconcat_s
        if cur[self.content].startswith("我们") and "继续" in cur[self.content]:
            disconcat_s = True
            return disconcat_s
        if cur[self.content].startswith("再来") or cur[self.content].startswith("没话找话"):
            disconcat_s = True
            return disconcat_s
        if cur[self.content].startswith("来看国际方面") or cur[self.content].startswith("市民大报料"):
            disconcat_s = True
            return disconcat_s
        if pre[self.content].endswith("详细内容。") or pre[self.content].startswith("观众朋友"):
            disconcat_s=True
            return disconcat_s
        if pre[self.content].endswith("来解决一些国际上的热点问题。"):
            disconcat_s=True
            return disconcat_s
        if pre[self.content].startswith("好") and "广告之后" in pre[self.content]: 
            disconcat_s=True
            return disconcat_s
        if cur[self.content].startswith("广告回来") or cur[self.content].startswith("观众朋友") or pre[self.content].startswith("观众朋友"):
            disconcat_s=True
            return disconcat_s
        if cur[self.content].startswith("夜线约见") or cur[self.content].startswith("来看今天的"):
            disconcat_s=True
            return disconcat_s
        return disconcat_s

    # 文本合并条件
    def get_concat_condition(self):
        with open(self.filename, encoding='utf-8') as f:
            data_all = json.load(f)
            # 读取json文件

            #将字典的值转为列表
            data_list = list(data_all.values())

        # ---------------- 判断条件1:如果两个分段音频的间隔时间大于给定阈值，则合并----------------------------

            index = 0
            pre_index = 0
            while index < len(data_list):
            
                news = data_list[index]
                # while news[self.content][-1]=='，' and data_list[next_index][self.content]!='嗯':
                #     self.concat_text(index,data_list,next_index)

                # if next_index<=len(data_list):
                #     index+=1
                #     next_index=index+1
                
                #删除开始新闻之前的所有内容
                if index!=0 and self.del_forward(news):
                    i=index
                    while i>=0:
                        data_list.pop(0)
                        i-=1
                    index=0
                    pre_index=0
            
                cur_start = news[self.start]
                pre_end = data_list[pre_index][self.end]

                dur_time = int(cur_start) - int(pre_end)


                #不合并，跳过
                if index != 0 and self.word_disconcat(data_list[pre_index],
                                                      news):
                    pre_index = index
                    index += 1
                    continue

                #合并
                if (dur_time < 500 or self.word_concat(data_list[pre_index],
                                                       news)) and index != 0:
                    self.concat_text(pre_index, data_list, index)
                else:
                    pre_index = index
                    if index <= len(data_list):
                        index += 1

            print("判断条件1:如果两个分段音频的间隔时间大于给定阈值，则合并")

            # -------------- 判断条件2:如果两个分段的文本相似度大于给定阈值，则合并--------------------------
            #self.stride_similarity_cal(20,data_list,0.65)
            pre_index = 0
            index = 0
            while index < len(data_list):
                news = data_list[index]

                #不合并，跳过
                if (index != 0 and self.word_disconcat(data_list[pre_index],news)) or index==0:
                    pre_index = index
                    index += 1
                    continue

                # 合并条件
                if index != 0 and not self.time_judge(data_list[pre_index], news) and (self.word_concat(data_list[pre_index], news) and float(self.text_similarity_pre(news[self.content], data_list[pre_index][self.content]) > 0.5)):
                    self.concat_text(pre_index, data_list, index)
                elif index!=0 and self.time_judge(data_list[pre_index], news) and float(self.text_similarity_pre(news[self.content], data_list[pre_index][self.content]) > 0.7):
                    self.concat_text(pre_index, data_list, index)
                else:
                    pre_index = index
                    if index <= len(data_list):
                        index += 1

        # data_all={}
        # for index,item in enumerate(data_list):
        #     data_all[index]=item
        # #保存json文件
        # self.save_json(data_all)
            print("判断条件2:如果两个分段的文本相似度大于给定阈值，则合并")

            #         # -------------- 判断条件3:如果本分段内有标题，则合并-------------------------
            #             pre_index = 0
            #             for index in list(data_list):
            #                 # 去除嗯嗯..
            #                 if data_list[index][self.content].count('嗯') >= 3:
            #                     continue
            #                 news = data_list[index]
            #                 video_start = int(news[self.start])
            #                 video_end = int(news[self.end])

            #                 cap = cv2.VideoCapture(self.video_file)  # 返回一个capture对象
            #                 # 记录本段视频出现标题的次数
            #                 tag = 0
            #                 for i in range(video_start, video_end):
            #                     cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # 设置要获取的帧号
            #                     ret, frame = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
            #                     if ret:
            #                         title_exit = judge_title(cap, frame)  # 判断是否存在标题区域
            #                     if title_exit:
            #                         tag += 1
            #                         # 标题出现的次数大于5，则认定为存在标题区域
            #                         if tag >= 5:
            #                             break

            #                 print(tag)
            #                 # 分段合并
            #                 if index != 0 and tag >= 1:
            #                     self.concat_text(pre_index, data_list, index)
            #                 pre_index = index

            #             print(data_list)

            #         with open(self.save_path, encoding='utf-8') as f:
            #             data_list = json.load("data/res_new_东方午新闻20140116.json")
            #             # 读取json文件
            #             #data_list=list(data_all)
            
        # -------------- 判断条件4-1:如果本分段和上一个段落关键词有重合且重合个数大于阈值，则合并-------------------------
            pre_index = 0
            index = 0
            while index < len(data_list):

                news = data_list[index]

                #不合并，跳过
                if index != 0 and self.word_disconcat(data_list[pre_index],
                                                      news):
                    pre_index = index
                    index += 1
                    continue

                cur_content = news[self.content]
                pre_content = data_list[pre_index][self.content]

                # 分段合并
                if index != 0 and (tag_extract(cur_content, pre_content,10,2)
                                   or self.word_concat(data_list[pre_index],
                                                       news)):
                    self.concat_text(pre_index, data_list, index)
                else:
                    pre_index = index
                    if index <= len(data_list):
                        index += 1

            print("判断条件4:如果本分段和上一个段落关键词有重合且重合个数大于阈值，则合并")

        # -------------- 判断条件4-2:如果本分段和上一个段落关键词有重合且重合个数大于阈值，则合并-------------------------
            pre_index = 0
            index = 0
            while index < len(data_list):
                cur_content = data_list[index][self.content]
                news = data_list[index]
                
                #不合并，跳过
                if index != 0 and self.word_disconcat(data_list[pre_index],
                                                      news):
                    pre_index = index
                    index += 1
                    continue

                cur_content = news[self.content]
                pre_content = data_list[pre_index][self.content]

                # 分段合并
                if index != 0 and (tag_extract(cur_content, pre_content,20,2)
                                   or self.word_concat(data_list[pre_index],
                                                       news)):
                    self.concat_text(pre_index, data_list, index)
                else:
                    pre_index = index
                    if index <= len(data_list):
                        index += 1

            print("判断条件4-2:如果本分段和上一个段落关键词有重合且重合个数大于阈值，则合并")



        # -------------- 判断条件4-3:如果本分段和上一个段落关键词有重合且重合个数大于阈值，则合并-------------------------
            pre_index = 0
            index = 0
            weather_begin=0
            while index < len(data_list):
                cur_content = data_list[index][self.content]
                news = data_list[index]

                 #删除所有嗯
                if news[self.content][0] == '嗯' and len(news[self.content])<5:
                    data_list.pop(index)
                    continue

                # if index>=weather_begin and not news[index].startswith("嗯"):
                #     weather_begin=index
                #     continue

               

                #不合并，跳过
                if index != 0 and self.word_disconcat(data_list[pre_index],
                                                      news):
                    pre_index = index
                    index += 1
                    continue
                
                
                cur_content = news[self.content]
                pre_content = data_list[pre_index][self.content]

            
                # 分段合并
                if index != 0 and (tag_extract(cur_content, pre_content,30,3)
                                   or self.word_concat(data_list[pre_index],
                                                       news)):
                    self.concat_text(pre_index, data_list, index)
                else:  
                    pre_index = index
                    if index <= len(data_list):
                        index += 1

            print("判断条件4-3:如果本分段和上一个段落关键词有重合且重合个数大于阈值，则合并")

         # ---------------- 判断条件5:关键词合并----------------------------

            index = 0
            pre_index = 0
            while index < len(data_list):
            
                news = data_list[index]

                  #删除所有嗯
                if news[self.content][0] == '嗯' and len(news[self.content])<5:
                    data_list.pop(index)
                    continue

                if self.weather_judge(news):
                    weather_begin=index+1

                #不合并，跳过
                if index != 0 and self.word_disconcat(data_list[pre_index],
                                                      news):
                    pre_index = index
                    index += 1
                    continue
                
                cur_content = news[self.content]
                pre_content = data_list[pre_index][self.content]
                #合并
                if index!=0 and (tag_extract(cur_content, pre_content,15,2,200) or self.word_concat(data_list[pre_index],news) or float(self.text_similarity_pre(news[self.content], data_list[pre_index][self.content]) > 0.69)):
                    self.concat_text(pre_index, data_list, index)
                else:
                    pre_index = index
                    if index <= len(data_list):
                        index += 1

            print("判断条件5:根据合并词合并")
        

        #合并天气预报
        self.concat_weather(data_list,weather_begin)
        #删除中间不需要的内容
        self.del_middle_content(data_list)
        #合并被嗯分割的同一新闻
        #self.concat_split_news(data_list)
        
        #删除其他部分
        self.del_other(data_list)
       
        #列表内容处理
        self.list_process(data_list)
           
        data_all = {}
        data_all["answer"] = data_list
        #保存json文件
        self.save_json(data_all)


#         # 保存结果
#         res = json.dumps(data_list, indent=4, ensure_ascii=False)
#         f_res = open("data/res_new.json", "w", encoding='utf8')
#         f_res.write(res)

if __name__ == "__main__":
    Text().get_concat_condition()
  