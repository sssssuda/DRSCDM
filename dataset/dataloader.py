from pathlib import Path

from pyannote.core import Annotation, Segment
from torch.utils.data import dataloader, dataset


def rttm_to_annotation(filename, target=None):
    annotation = Annotation()
    # vad_hypothesis = VAD("D:\\fyh\\design\\SDsys\\amicorpus\\EN2001a\\audio\\EN2001a.Mix-Headset.wav")

    with open(filename, "r") as f:
        for line in f.readlines():
            detail = line.strip().split(" ")
            if target:
                if detail[1] != target:
                    continue
            segment = Segment(float(detail[3]), float(detail[3]) + float(detail[4]))
            annotation[segment, "_"] = detail[-3]

    # print(f"number of speakers:{len(annotation.labels())}")
    # print(annotation)
    return annotation


def get_data(dataset, subset="dev"):
    """
        data_list: 数据集划分的路径--文件名
        data_path: 数据集文件的路径--文件
        rttm_path: rttm文件路径--标注

        self.data = [{"uri": filename, "audio": "path/to/file", "annotation": Annotation()}]
    """

    datas = []
    if dataset == "AMI":
        data_list = "d:/FYH/datas/AMI-diarization-setup/lists"
        data_path = "d:/FYH/datas/amicorpus"
        rttm_path = "d:/FYH/datas/AMI-diarization-setup/word_and_vocalsounds/rttms"

        data_list = data_list + "/" + subset + ".meetings.txt"
        lines = open(data_list).read().splitlines()
        for filename in lines:
            data = {"uri": filename, "audio": Path(f"{data_path}/{filename}/audio/{filename}.Mix-Headset.wav"),
                    "annotation": rttm_to_annotation(f"{rttm_path}/{subset}/{filename}.rttm", target=None)}
            datas.append(data)

    elif dataset == "VoxConverse":
        data_list = "d:/FYH/datas/voxconverse"
        data_path = f"d:/FYH/datas/voxconverse_{subset}_wav"
        rttm_path = "d:/FYH/datas/voxconverse"

        data_list = data_list + "/lists/" + subset + ".txt"
        lines = open(data_list).read().splitlines()
        for filename in lines:
            data = {"uri": filename, "audio": Path(f"{data_path}/audio/{filename}.wav"),
                    "annotation": rttm_to_annotation(f"{rttm_path}/{subset}/{filename}.rttm", target=None)}
            datas.append(data)

    return datas

