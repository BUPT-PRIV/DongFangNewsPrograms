from silence_seg import SilenceSeg
from pydub import AudioSegment
import os
import sys
#sys.path.append("../")
from audio_rec import AudioRec
from moviepy.editor import *
from pydub import AudioSegment


class AudioToText:
    def __init__(self) -> None:
        self.load_audio="result/"
        self.audio_name="test.wav"
    
    def audio_trans(self,video_name):
        
        #读取视频转为wav格式音频
        video = VideoFileClip(video_name)
        audio = video.audio
        audio.write_audiofile(self.audio_name)
        sound = AudioSegment.from_mp3(self.audio_name)
        #静音分段
        SilenceSeg().get_audio_seg(sound)

    def save_transfer_text(self):
        files=os.listdir(self.load_audio)
        files_num=len(files)
       
        for i in range(files_num):
            #读取分段后的音频并识别
            address_now=self.load_audio+"audio"+str(i)+".wav"
            print(i,AudioRec().test_one(address_now))

def test_video(video_name):
    AudioToText().audio_trans(video_name)
    AudioToText().save_transfer_text()

test_video("test.mp4")

        

