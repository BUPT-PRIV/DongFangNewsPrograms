import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from moviepy.editor import AudioFileClip

class AudioRec(object):
    def __init__(self):
        super(AudioRec).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
        self.model = Wav2Vec2ForCTC.from_pretrained("ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt")
        self.resampler = torchaudio.transforms.Resample(48_000, 16_000)
        self.model.to("cuda")

    def speech_file_to_array_fn(self,place):
        speech_array, sampling_rate = torchaudio.load(place)
        speech_array = self.resampler(speech_array[0,:]).squeeze().numpy()
        return speech_array

    def test_one(self,place):
        speech = self.speech_file_to_array_fn(place)
        speech = self.processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(speech.input_values.to("cuda"), attention_mask=speech.attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)
