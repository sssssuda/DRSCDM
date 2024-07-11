"""
    bbt_s01e01.npy: 文件中存储该影集中按顺序出现的人像embeddings，但不对应到相应的帧（即没有具体出现时间戳）
    bbt_s01e01_list.npy: 文件中存储该影集按顺序出现的track id，无人物标签

"""


import glob
import sys

import numpy
import tqdm

# id.npy, 包含每个id的出现次数的embeds
trackroot_path = "D:/FYH/datas/bbt_buffy/tracks"
idroot_path = "D:/FYH/datas/bbt_buffy/features"
saveroot_path = "D:/FYH/datas/bbt_buffy/streams_features"
spkroot_path = "D:/FYH/datas/bbt_buffy/ids"
file_list = ["bbt_s01e01", "bbt_s01e02", "bbt_s01e03", "bbt_s01e04", "bbt_s01e05", "bbt_s01e06", "buffy_s05e01",
             "buffy_s05e02", "buffy_s05e03", "buffy_s05e04", "buffy_s05e05", "buffy_s05e06"]


def get_episode_spks_embeds(root_path):
    id2embeds = {}
    id_files = glob.glob(f"{root_path}/*.npy")
    for filename in id_files:
        filename = filename.replace("\\", "/")
        id = int(filename.split("/")[-1].split(".")[0])
        embeds = numpy.load(filename)
        id2embeds[id] = embeds

    return id2embeds


def get_spk2track_dict(path):
    spk2track = {}
    with open(path, "r") as f:
        i = 1
        for line in f.readlines():
            if i > 0:
                i -= 1
                continue
            spk = line.strip().split(" ")[1]
            if spk not in spk2track:
                spk2track[spk] = []
            spk2track[spk].append(int(line.strip().split(" ")[0]))

    return spk2track


def get_episode_stream_embeds(filename):
    npy_path = idroot_path + f"/{filename}"
    id2embeds = get_episode_spks_embeds(npy_path)
    spk2track_path = spkroot_path + f"/{filename}.ids"
    spk2track = get_spk2track_dict(spk2track_path)
    tracks_path = trackroot_path + f"/{filename}.tracks"
    stream_embeds = numpy.zeros((1, 256))
    with open(tracks_path, "r") as f:
        i = 2
        stream_spks = []
        for line in tqdm.tqdm(f.readlines(), desc=f"File:{filename}, Readlines"):
            if i > 0:
                i -= 1
                continue

            detail = line.strip().split(" ")
            for i in range(int(detail[2])):
                track_id = int(detail[5 * i + 3])
                for spk, tracks in spk2track.items():
                    if track_id not in tracks: continue
                    stream_spks.append(spk)
                    break

                embed = id2embeds[track_id][0]
                embed = numpy.reshape(embed, (1, 256))
                id2embeds[track_id] = numpy.delete(id2embeds[track_id], 0, axis=0)
                stream_embeds = numpy.concatenate((stream_embeds, embed), axis=0)
            sys.stdout.flush()

    return stream_embeds[1:], numpy.array(stream_spks)


def prepare_face_data():
    for filename in file_list:
        streams, labels = get_episode_stream_embeds(filename)
        print(streams.shape)
        numpy.save(f"{saveroot_path}/{filename}.npy", streams)
        numpy.save(f"{saveroot_path}/{filename}_list.npy", labels)

# a = numpy.load("D:/FYH/datas/bbt_buffy/streams_features/bbt_s01e06_list.npy")
# b = numpy.load("D:/FYH/datas/bbt_buffy/streams_features/bbt_s01e06.npy")
# print(a.shape)
# print(b.shape)


def get_face_data():
    datas = []
    for filename in file_list:
        streams_embedings = numpy.load(f"{saveroot_path}/{filename}.npy")
        streams_labels = numpy.load(f"{saveroot_path}/{filename}_list.npy")
        data = {"uri": filename, "features": streams_embedings, "labels":streams_labels}
        datas.append(data)

    return datas
