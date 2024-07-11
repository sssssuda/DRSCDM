import time
from functools import partial

import numpy as np
import torch
import tqdm
import torch.nn.functional as F
from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection
from pyannote.core import Segment, Timeline, Annotation
from pyannote.audio import Model
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.diarization import DiarizationErrorRate
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

from Cluster import DensityClustering, CosClustering, DataPoint, DataPoint1, BaseClustering
from VBx.vbhmm import vbhmm1
from dataset import get_data, get_face_data
from embedding_models import ECAPA_TDNN
from embedding_models.extraction import split_segments, extract_embeddings_wav
from extract_audio import preprocessing, post_processing

model_path = "./pretrained/ECAPA_pretrain.model"
model_path1 = "d:/FYH/ECAPA-TDNN/exps/exp1/model1/model_0010.model"


def init_model(dataset):
    # audio_stream = preprocessing("./data/VoxConverse/ccokr.wav")
    embed_model = ECAPA_TDNN(C=1024).cuda()
    embed_model.load_parameters(model_path)
    # embed_model = None

    if dataset == "AMI":
        VAD_HYPER_PARAMETERS = {
            # onset/offset activation thresholds
            "onset": 0.684, "offset": 0.577,
            # remove speech regions shorter than that many seconds.
            "min_duration_on": 0.181,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": 0.037
        }
    elif dataset == "VoxConverse":
        VAD_HYPER_PARAMETERS = {
            # onset/offset activation thresholds
            "onset": 0.767, "offset": 0.713,
            # remove speech regions shorter than that many seconds.
            "min_duration_on": 0.182,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": 0.501
        }

    vad_osd_joint = Model.from_pretrained("./pretrained/vad_model.bin")

    VAD = VoiceActivityDetection(segmentation=vad_osd_joint)
    VAD.instantiate(VAD_HYPER_PARAMETERS)
    # OSD = OverlappedSpeechDetection(segmentation=vad_osd_joint)
    # OSD.instantiate(OSD_HYPER_PARAMETERS)
    # VAD = None

    return VAD, embed_model.eval()


def cosine(x, y, eps=1e-15):
    x = x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps)
    y = y / np.linalg.norm(y, axis=-1, keepdims=True).clip(min=eps)
    return np.dot(x, y.T)


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

    print(f"number of speakers:{len(annotation.labels())}")
    # print(annotation)
    return annotation


def cluster_to_annotation(clusters):
    annotation = Annotation()
    for cluster in clusters:
        for segment in cluster.timeline:
            annotation[segment, "_"] = cluster.name

    # print(annotation)
    return annotation


def cluster_to_annotation1(clusters):
    annotation = Annotation()
    for cluster, timeline in clusters.items():
        for segment in timeline:
            annotation[segment, "_"] = cluster

    # print(annotation)
    return annotation


# data_stream = torch.tensor([[0.6, 0.8], [0.59, 0.79], [0.61, 0.81], [0.1, 0.2], [0.09, 0.19], [0.11, 0.21]])
def density_cluster(files):
    t_offline = 30
    eps_min, eps_max, min_weight, decay_factor, threshold_w = -0.2, -0.1, 10, 0.002, 0.1

    for idx, file in tqdm.tqdm(enumerate(files), total=len(files)):
        audio_streams = preprocessing(file["audio"])
        reference = file["annotation"]
        reference.uri = file["uri"]
        cluster = DensityClustering(eps_min, eps_max, min_weight, decay_factor, threshold_w)

    for i, segment in enumerate(audio_stream):  # segment: 0.1s
        vad = True  # VAD(segment)    # true: speech, false: noise/silence
        if vad:
            embeddings = segment  # EMB(segment)  # low-dimension
            p = DataPoint(i, embeddings)  # data point need to cluster
            cluster.add(p)
        if (i + 1) % t_offline == 0:
            cluster.updatecluster(i)
            result = cluster.result()

            result = post_processing(result)  # 合并连续时间
            print(result)


def online_cluster(files, VAD, embed_model, CLUSTERING="cos"):
    t, duration, timestamp, slide = 0, 4, 0, 2
    step = 16000 * slide

    """ 
    从音频流 audio_stream 中以4s的滑动窗口，每次移动2s，对窗口内的音频检测vad_timeline，接受这个时延
    不接受时长短于0.2s的语音段；当末尾的语音段长度超过0.2s但短于1s时，认为此语音段被截断了，continue到下一个滑动窗口
    """

    # metric2 = DetectionErrorRate(collar=0.25, skip_overlap=True)
    metric = DiarizationErrorRate(collar=0.25, skip_overlap=True)
    for minpts in [2]:
        for maxpts in [6]:
            start = time.time()
            a = 0
            for idx, file in tqdm.tqdm(enumerate(files), total=len(files)):
                audio_streams = preprocessing(file["audio"])
                reference = file["annotation"]
                reference.uri = file["uri"]
                # n_clusters = len(reference.labels())

                if CLUSTERING == "cos":
                    cluster = CosClustering(alpha=60, beta=65, min_segments=10, minpts=minpts,
                                            maxpts=maxpts)
                elif CLUSTERING == "density":
                    if dataset == "VoxConverse":
                        cluster = DensityClustering(eps_min=8, eps_max=10, min_weight=1, decay_factor=0.004,
                                                    threshold_w=0.1)  # 27分半钟前的数据消亡
                    else:
                        cluster = DensityClustering(eps_min=8, eps_max=10, min_weight=5, decay_factor=0.002,
                                                    threshold_w=0.1)  # 27分半钟前的数据消亡
                elif CLUSTERING == "baseline":
                    cluster = BaseClustering(threshold=0.65)

                t, timestamp, stop = 0, 0, False
                while not stop:
                    if len(audio_streams) > (16000 * duration + step * t):
                        waveforms = audio_streams[step * t: 16000 * duration + step * t]
                    else:
                        waveforms = audio_streams[step * t:]
                        waveforms = torch.nn.functional.pad(waveforms,
                                                            (0, 16000 * duration - waveforms.size(0)))
                        stop = True
                    current_file = {"waveform": waveforms.unsqueeze(0), "sample_rate": 16000,
                                    "uri": file["uri"]}
                    # vad_reference = reference.get_timeline().support().crop(Segment(t * slide, t * slide + duration))
                    vad_timeline = VAD(current_file).get_timeline().support()

                    for segment in vad_timeline:
                        if (segment.start + t * slide) >= timestamp:
                            if (segment.end >= duration - 0.06) and (segment.duration <= 1) and (not stop):
                                continue
                            with torch.no_grad():
                                wave = waveforms[
                                       int(segment.start * 16000) - 120:int(segment.end * 16000) + 680]
                                embed = embed_model(wave.unsqueeze(0).cuda())
                            # real_segment = seg
                            real_segment = Segment(segment.start + t * slide - 0.008,
                                                   segment.end + t * slide + 0.043)

                            p = DataPoint1(real_segment, embed.cpu())
                            clusters = cluster.add(p, stop=stop)

                            timestamp = segment.end + t * slide  # 记录保存的最后时间点

                    t += 1

                # file = {"waveform": audio_streams.unsqueeze(0), "sample_rate": 16000, "uri": 1}
                # hypothesis = VAD(file)
                # print(f"emb_time:{emb_time}\nclustering_time:{clustering_time}")
                hypothesis = cluster_to_annotation1(clusters)
                # with open(f"./data/rttm/{file['uri']}.rttm", "w") as f:
                #     hypothesis.write_rttm(f)

                _ = metric(reference=reference, hypothesis=hypothesis, detailed=False)
                # metric2(reference=reference, hypothesis=hypothesis, detailed=True)
            end = time.time()
            avg_runtime = (end - start) / len(files)
            der = metric.report().iloc[-1, 0]
            print(f"Baseline, runtime:{avg_runtime}, der:{der}")
            print(der)
            with open("result.txt", "a") as f:
                f.write(f"Baseline, runtime:{avg_runtime}, der:{der}\n")

    return metric


def offline_clustering(files, VAD, embed_model, CLUSTERING="ahc"):
    metric = DiarizationErrorRate(collar=0.25, skip_overlap=True)
    for idx, file in tqdm.tqdm(enumerate(files), total=len(files)):
        if file["uri"] == "vylyk": continue
        audio_streams = preprocessing(file["audio"])
        current_file = {"waveform": audio_streams.unsqueeze(0), "sample_rate": 16000, "uri": file["uri"]}
        vad_annotation = VAD(current_file)
        vad_timeline = vad_annotation.get_timeline().support()

        segments = split_segments(vad_timeline)
        embeddings = extract_embeddings_wav(embed_model, audio_streams, segments, batch_size=160)

        reference = file["annotation"]
        reference.uri = file["uri"]
        n_clusters = len(reference.labels())

        from sklearn.cluster import AgglomerativeClustering, SpectralClustering
        if CLUSTERING == "ahc":
            threshold = -0.05

            similarity_matrix = cosine(embeddings, embeddings)

            distance_matrix = (-1.0) * similarity_matrix
            distance_threshold = (-1.0) * threshold
            cluster_model = AgglomerativeClustering(n_clusters=None,
                                                    distance_threshold=distance_threshold,
                                                    compute_full_tree=True,
                                                    affinity="precomputed",
                                                    linkage="complete")
            y_pred = cluster_model.fit_predict(distance_matrix)
            # print(y_pred)

        elif CLUSTERING == "sc":
            if embeddings.shape[0] < 10:
                n_neighbors = embeddings.shape[0]
            else:
                n_neighbors = 10
            cluster_model = SpectralClustering(n_clusters=n_clusters,
                                               n_neighbors=n_neighbors,
                                               affinity="nearest_neighbors",
                                               assign_labels="kmeans")
            y_pred = cluster_model.fit_predict(embeddings)

        elif CLUSTERING == "sc1":
            """ not support """
            if embeddings.shape[0] < 10:
                n_neighbors = embeddings.shape[0]
            else:
                n_neighbors = 10

            S = cosine(embeddings, embeddings)
            sigma = 1.0
            k = n_neighbors
            N = S.shape[0]
            A = np.zeros((N, N))

            for i in range(N):
                dist_with_index = zip(S[i], range(N))
                dist_with_index = sorted(dist_with_index, key=lambda x: x[0])
                neighbours_id = [dist_with_index[m][1] for m in range(k + 1)]  # xi's k nearest neighbours
                for j in neighbours_id:  # xj is xi's neighbour
                    A[i][j] = np.exp(-S[i][j] / 2 / sigma / sigma)
                    A[j][i] = A[i][j]  # mutually

            degreeMatrix = np.sum(A, axis=1)

            # compute the Laplacian Matrix: L=D-A
            laplacianMatrix = np.diag(degreeMatrix) - A

            # normailze
            # D^(-1/2) L D^(-1/2)
            sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
            laplacian = np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
            lam, H = np.linalg.eig(laplacian)

            sp_kmeans = KMeans(n_clusters=n_clusters).fit(H)
            y_pred = sp_kmeans.labels_

        elif CLUSTERING == "ahc+vb":
            y_pred = vbhmm1(embeddings)

        hypothesis = Annotation()
        for i, seg in enumerate(segments):
            hypothesis[seg, "_"] = y_pred[i]

        _ = metric(reference=reference, hypothesis=hypothesis, detailed=True)

        # print(metric)
        # exit(0)

    return metric


if __name__ == '__main__':
    mode = "online"
    for dataset in ["AMI", "VoxConverse"]:
        VAD, embed_model = init_model(dataset)
        for subset in ["dev", "test"]:
            files = get_data(dataset=dataset, subset=subset)
            # files = get_data(dataset="VoxConverse", subset=subset)
            if mode == "offline":
                metric = offline_clustering(files, VAD, embed_model, CLUSTERING="sc")
            else:
                metric = online_cluster(files, VAD, embed_model, CLUSTERING="density")
            # print(f"Dataset:{dataset}.{subset}")
            print(metric)
