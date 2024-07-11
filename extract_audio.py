import librosa
import numpy as np
import torch
from pyannote.core import Segment


def preprocessing(filepath):  # , model, model_path):
    audio = librosa.load(filepath, sr=16000)
    audio = audio[0]
    # print(type(audio), audio.shape, len(audio))
    # model.load_parameters(model_path)
    # model.eval()

    # if len(audio) % time != 0:
    #     audio = np.pad(audio, (0, time - len(audio) % time), 'wrap')
    # audio = torch.FloatTensor(np.reshape(audio, (-1, time)))
    # with torch.no_grad():
    #     audio_stream = model(audio.cuda())
    #     audio_stream = F.normalize(audio_stream, p=2, dim=1)

    return torch.FloatTensor(audio)


def post_processing(clusters):
    result = {}
    for cluster, time_list in clusters.items():
        time_list.sort()
        result[cluster] = []
        start, end = time_list[0], time_list[0] + 1
        for current in time_list[1:]:
            if (current == end) and (current != time_list[-1]):  # 当前后是连续帧时，合并
                end = current + 1
            elif current != time_list[-1]:  # 不是连续帧时，保存前序片段时间，重置当前片段时间
                result[cluster].append(Segment(start * 0.1, end * 0.1))
                start, end = current, current + 1
            else:
                result[cluster].append(Segment(start * 0.1, (current + 1) * 0.1))

    return result
