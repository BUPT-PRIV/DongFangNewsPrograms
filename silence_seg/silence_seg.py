import pydub
import json
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os


class SilenceSeg:
    def __init__(self) -> None:
        self.silence_len=50
        self.s_thresh=-60
    
    def get_audio_seg(self,sound):
        loudness = sound.dBFS
        audio_length_second=sound.duration_seconds
        audio_length=len(sound)

        not_silence_ranges = pydub.silence.detect_nonsilent(sound, min_silence_len=self.silence_len,silence_thresh=self.s_thresh, seek_step=1)
        last_end_position = 0 # 上个非静默音频段结束位置，初始为0

        json_dict = {}
        sounds=[]

        print("not silence: ",len(not_silence_ranges))
        for index in range(len(not_silence_ranges)):
            json_dict2 = {}
            current_end_position = round((not_silence_ranges[index][1])) # 获取当前非静默音频段结束位置
            if index == len(not_silence_ranges)-1:
                new = sound[last_end_position:]
            else:
                new = sound[last_end_position:current_end_position]
            new.export("result/audio{0}.wav".format(index),format="wav")
            print(index)

            print(last_end_position,current_end_position)
    
            #保存至json文件
            json_dict2["StartPosition"] = last_end_position
            json_dict2["EndPosition"] = current_end_position
            json_dict2["Duration"] = int(new.duration_seconds*1000)

            json_dict[index] = json_dict2

            res = json.dumps(json_dict, indent=4, ensure_ascii=False)
            f_res = open(r"res.json", "w", encoding='utf8')
            f_res.write(res)

            last_end_position = current_end_position


#SilenceSeg().get_audio_seg()  
