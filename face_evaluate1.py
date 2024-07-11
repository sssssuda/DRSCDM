"""
Train model for face clustering
"""

import os
import sys
import glob

import numpy
import numpy as np
import torch
import tqdm

from facecluster import BaseClustering2, DensityClustering2, CosClustering2, DataPoint1, DataPoint2
from read_frame import get_episode_stream_embeds

torch.manual_seed(1111)

# Local imports
import face_metrics as metrics


# CPU / GPU
device = None


def cosine(x, y, eps=1e-15):
    x = x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps)
    y = y / np.linalg.norm(y, axis=-1, keepdims=True).clip(min=eps)
    return np.dot(x, y.T)


def cluster2list_cos(clusters, size):
    c_list = []
    for index in range(1, size + 1):
        for cluster in clusters:
            if index in cluster.idxes:
                c_list.append(int(cluster.name[1:]))
                break

    return numpy.array(c_list)


def cluster2list_base(clusters, size):
    c_list = []
    for index in range(1, size + 1):
        for name, idxes in clusters.items():
            if index in idxes:
                c_list.append(int(name[1:]))
                break
    return numpy.array(c_list)


def cluster2list_density(clusters, size):
    c_list = []
    for index in range(1, size + 1):
        for name, idxes in clusters.items():
            if index in idxes:
                c_list.append(int(name[1:]))
                break
    return numpy.array(c_list)


def validate(dset, curve=False):
    """Evaluate model performance
    """
    # evaluation dataset is simple
    X, y_gt = dset

    ### CLUSTERING ###
    print('Performing clustering')
    cluster = BaseClustering2(threshold=0.5)
    cluster = DensityClustering2(eps_min=-0.6, eps_max=-0.4, min_weight=1,
                                 decay_factor=0.004, threshold_w=0.1)
    for i, embed in tqdm.tqdm(enumerate(X, start=1), desc="frame clustering"):
        p = DataPoint1(i, torch.from_numpy(embed).unsqueeze(0))
        # TODO: VC TRSF, add a new DR MLP
        # p = MLP(p)

        clusters = cluster.add(p)
    C = cluster2list_base(clusters, y_gt.size)

    # from sklearn.cluster import AgglomerativeClustering, SpectralClustering
    # threshold = 0.1
    #
    # similarity_matrix = cosine(X, X)
    #
    # distance_matrix = (-1.0) * similarity_matrix + 1
    # distance_threshold = (-1.0) * threshold + 1
    # cluster_model = AgglomerativeClustering(n_clusters=None,
    #                                         distance_threshold=distance_threshold,
    #                                         compute_full_tree=True,
    #                                         affinity="precomputed",
    #                                         linkage="complete")
    # C = cluster_model.fit_predict(distance_matrix)

    ### CLUSTERING METRICS ###
    nT = C.size  # number of tracks
    # number of clusters
    nY = np.unique(y_gt).size  # numel
    nC = np.unique(C).size  # numel
    # metrics
    nmi = metrics.NMI(y_gt, C)
    wcp = metrics.weighted_purity(y_gt, C)[0]
    # print, store and return
    print('#Clusters in T: {:5d}, Y: {:4d}, C: {:4d}, NMI: {:.4f}, Purity: {:.4f}'.format(nT, nY, nC, nmi, wcp))
    val_metrics = {'nmi': nmi, 'wcp': wcp, 'nY': nY, 'nC': nC}
    with open("face_result.txt", "a") as f:
        f.write('#Clusters in T: {:5d}, Y: {:4d}, C: {:4d}, NMI: {:.4f}, Purity: {:.4f}\n'.format(nT, nY, nC, nmi, wcp))

    # return packaging
    return_things = [val_metrics]
    if curve:
        # purity curve
        curves = hac.evaluate_curve(y_gt, Z, 200, curve_metrics=['wcp', 'nmi'])
        return_things.append(curves)

    return return_things


def simple_read_dataset(video):
    """Simple dataset reading function for purpose of checking evaluation code
    """

    print('Loading dataset:', video)
    with open("face_result.txt", "a") as f:
        f.write(f'Loading dataset:{video}\n')

    # Read label file
    label_fname = 'd:/FYH/datas/bbt_buffy/ids/' + video + '.ids'
    with open(label_fname, 'r') as fid:
        fid.readline()  # ignore header
        data = fid.readlines()  # track to name
        data = [line.strip().split() for line in data if line.strip()]
        # trackid --> name mapping
        ids = {int(line[0]): line[1] for line in data}

    # get unique names and assign numbers
    uniq_names = list(set(ids.values()))

    # Read feature files
    X, y = [], []
    all_feature_fname = glob.glob('d:/FYH/datas/bbt_buffy/features/' + video + '/*.npy')
    for fname in all_feature_fname:
        # load and append feature
        feat = np.load(fname)
        X.append(feat.mean(0))
        # append label
        tid = int(os.path.splitext(os.path.basename(fname))[0])
        y.append(uniq_names.index(ids[tid]))

    X = np.array(X)
    y = np.array(y)
    return [X, y]


def read_dataset_frame(video):
    print('Loading dataset:', video)

    stream_embeds, labels = get_episode_stream_embeds(video)
    return stream_embeds, labels


def main(video):
    """Main function
    """
    gpu = 0
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu != -1 else "cpu")
    print(device)

    ### Dataset ###
    # simplified evaluation example (normally uses PyTorch datasets)
    X, y = simple_read_dataset(video)


    ### Run evaluation ###
    val_metrics = validate([X, y], curve=False)
    print(val_metrics)


valid_videos = ['bbt_s01e01', 'bbt_s01e02', 'bbt_s01e03', 'bbt_s01e04', 'bbt_s01e05', 'bbt_s01e06',
                'buffy_s05e01', 'buffy_s05e02', 'buffy_s05e03', 'buffy_s05e04', 'buffy_s05e05', 'buffy_s05e06']

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1]:
        video = sys.argv[1]
    else:
        video = 'bbt_s01e01'
    assert video in valid_videos, 'Erroneous video name. Valid videos: {}'.format(valid_videos)

    for video in valid_videos:
        main(video)

    # main("bbt_s01e06")
