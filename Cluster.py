import math
import numpy as np
import torch

from numpy import sort
from pyannote.core import Segment, Timeline


class DataPoint:
    def __init__(self, segment, embed):
        self.segment = segment
        self.embed = embed


class DataPoint1:
    def __init__(self, segment, embed, kernel=None, shell=None, membership=None):
        if kernel is None:
            kernel = set()
        if shell is None:
            shell = set()
        if membership is None:
            membership = {}

        self.timeline = Timeline([segment])
        self.end = segment.end
        self.embed = embed

        self.kernel = kernel
        self.shell = shell
        self.membership = membership

    def merge(self, data):
        """ 把 data 合并到 self 中 """
        self.timeline = self.timeline.union(data.timeline).support()
        self.end = max(self.end, data.end)

        del data
        return self

    def weight(self, t, factor):
        """ 在t时刻，self点的权重 """
        return pow(2, -1.0 * factor * (t - self.end))


class DataPoint2:
    def __init__(self, index, embed):
        self.index = [index]
        self.embed = embed


class Cluster:
    def __init__(self, name):
        self.timeline = Timeline()
        self.name = name

    def add(self, seg: Segment):
        self.timeline.add(seg)
        self.timeline = self.timeline.support()


class Cluster1:
    def __init__(self, name, p: DataPoint):
        self.point = set()
        self.name = name

        self.point.add(p)

    def add(self, q: DataPoint):
        self.point.add(q)


class Cluster2:
    def __init__(self, name):
        self.idxes = []
        self.name = name

    def add(self, index: int):
        self.idxes += index


class BaseClustering:
    def __init__(self, threshold):
        self.thrs = threshold
        self.centroids = None
        self.clusters = None
        self.labels = []

    def add(self, p):
        if self.clusters is None:
            cluster = {"name": "c1", "embeds": p.embed, "centroid": p.embed, "timeline": Timeline([p.segment])}
            self.clusters = [cluster]
            self.centroids = p.embed
            return self.result()

        sim_matrix = self.cosine_similarity_matrix(p.embed, self.centroids)
        max_index = int(torch.argmax(sim_matrix)) + 1
        if sim_matrix[0][max_index - 1] >= self.thrs:
            self.labels.append(max_index)
            for i, cluster in enumerate(self.clusters, start=1):
                if i == max_index:
                    cluster["embeds"] = torch.cat([cluster["embeds"], p.embed], dim=0)
                    cluster["centroid"] = self.calculate_centroid(cluster["embeds"])
                    self.centroids[i - 1] = cluster["centroid"]
                    cluster["timeline"] = cluster["timeline"].add(p.segment)
        else:
            cluster = {"name": f"c{len(self.clusters) + 1}", "embeds": p.embed,
                       "centroid": p.embed, "timeline": Timeline([p.segment])}
            self.clusters.append(cluster)
            self.centroids = torch.cat([self.centroids, p.embed], dim=0)

        return self.result()

    def cosine_similarity_matrix(self, x, y):
        x = x / torch.linalg.norm(x, dim=1, keepdim=True)
        y = y / torch.linalg.norm(y, dim=1, keepdim=True)

        return torch.matmul(x, y.T)

    def calculate_centroid(self, embeds):
        centroid = embeds[0]
        size = embeds.shape[0]
        for embed in embeds[1:]:
            centroid = centroid + embed

        return centroid / size

    def result(self):
        result = {}
        for cluster in self.clusters:
            result[cluster["name"]] = cluster["timeline"]

        return result


class DensityClustering:
    def __init__(self, eps_min, eps_max, min_weight, decay_factor, threshold_w):
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.min_weight = min_weight
        self.decay_factor = decay_factor
        self.threshold_w = threshold_w
        self.pList = set()
        self.res = {}
        self.t = 0

    def add(self, p: DataPoint1, stop=False):
        """ 将新到达的数据点p 加入到列表中 """

        p_kernel, p_shell = self.fuzzyquery(p, self.pList)
        for k in p_kernel:
            k.kernel.add(p)
        for s in p_shell:
            s.shell.add(p)
        self.pList.add(p)
        self.t += 1

        if self.t % 5 == 0: self.updatecluster(p.end)
        return self.result()

    def fuzzyquery(self, p, pList):
        """ 模糊查询，找到数据点p kernel/shell区域的数据点 """

        for q in pList:
            distance = self.distance(p.embed, q.embed)
            if distance <= self.eps_min:
                p.kernel.add(q)
            elif distance <= self.eps_max:
                p.shell.add(q)

        return p.kernel, p.shell

    def updatecluster(self, time):
        """ 1.删除过期点并合并 2.寻找核心点并扩展 3.计算边界点的隶属度 """

        threshold_t = math.log2(self.threshold_w) / self.decay_factor + time
        remove = []
        for p in self.pList:
            t = p.end
            if t < threshold_t:
                remove.append(p)

                near_p = self.findNearest(p)
                near_p.merge(p)
        for p in remove:
            self.pList.remove(p)

        visited = set()
        # TODO: 如何保证在两次的offline中，类别顺序相同-排列不一致问题
        clusters = set()
        for p in self.pList:
            to_visit = set()
            first = True
            if p not in visited:
                to_visit.add(p)

                flag = True
                while flag:  # for q in to_visit:
                    if to_visit:
                        q = to_visit.pop()
                    else:
                        flag = False
                        continue
                    if self.evaluateCore(q, time):
                        if q not in visited:
                            visited.add(q)
                            if first:
                                index = len(clusters)
                                cluster = Cluster1(f"C{index + 1}", q)
                                first = False
                                clusters.add(cluster)
                            cluster.add(q)

                            to_visit = to_visit | q.kernel
                            for s in q.kernel | q.shell:
                                # if not self.isCore(s, t):
                                self.fuzzyborder(s, q, clusters)

        # print(f"聚类成{len(clusters)}个簇！")

    def evaluateCore(self, p, time):
        """ 1.剔除数据点区域内的过期点 2.计算是否是核心点 """
        threshold_t = math.log2(self.threshold_w) / self.decay_factor + time
        sum_w = p.weight(time, self.decay_factor)
        remove = []
        for s in p.shell:
            t = s.end
            if t < threshold_t:
                remove.append(s)
        for s in remove:
            p.shell.remove(s)
        remove = []
        for k in p.kernel:
            t = k.end
            if t < threshold_t:
                remove.append(k)
            else:
                sum_w += k.weight(time, self.decay_factor)
        for k in remove:
            p.kernel.remove(k)
        if sum_w >= self.min_weight:
            return True
        return False

    def isCore(self, p, t):
        """ 仅计算是否是核心点 """
        sum_w = p.weight(t, self.decay_factor)
        for k in p.kernel:
            sum_w += k.weight(t, self.decay_factor)
        if sum_w >= self.min_weight:
            return True
        return False

    def fuzzyborder(self, s, p, clusters):
        """ 模糊边界：计算隶属度 """
        membership_degree = self.fuzzymembership(s, p)
        for cluster in clusters:
            if p in cluster.point:
                s.membership[cluster.name] = max(s.membership.get(cluster.name, 0), membership_degree)

    def fuzzymembership(self, s, p):
        """ min内的数据是1，max内的数据按距离计算 """
        distance = self.distance(s.embed, p.embed)
        if distance < self.eps_min:
            return 1
        else:
            return (self.eps_max - distance) / (self.eps_max - self.eps_min)

    def result(self):
        """ 更新内容，计算结果 """
        result = {}
        hold_list = []
        for p in self.pList:
            # print(f"{p.embed}: {p.membership}")
            if p.membership:
                label = max(p.membership, key=p.membership.get)
                p.membership = {}
            else:
                hold_list.append(p)
                continue
            if label not in result:
                result[label] = p.timeline
            else:
                result[label] = result[label].union(p.timeline).support()

            self.res = result

        if hold_list:
            cname = f"c{len(result) + 1}"
            result[cname] = hold_list[0].timeline
            for p in hold_list[1:]:
                result[cname] = result[cname].union(p.timeline).support()

        return self.res

    def findNearest(self, p):
        """ 找到离当前点最近的数据点 """
        nearest = None
        min_distance = math.inf
        for k in p.kernel:
            dist = self.distance(k.embed, p.embed)
            if dist < min_distance:
                min_distance = dist
                nearest = k

        if nearest:
            return nearest
        return p

    def distance(self, embed1, embed2):
        """ 欧几里得距离 """
        euclidean_distance = torch.nn.functional.pairwise_distance(embed1.unsqueeze(0), embed2.unsqueeze(0))
        # score1 = torch.matmul(embed1, embed2.T)
        # score = torch.mean(torch.matmul(embed1, embed2.T)) * -1
        return euclidean_distance  # score.detach().cpu().numpy()    #


class CosClustering:
    """
        pList: [p1, p2, ..., pn]
        clusters: [Cluster1, Cluster2, ..., Clustern]
        center_vectors: {"cv1": {"embed": tensor, "}}
    """

    def __init__(self, alpha=55, beta=60, min_segments=10, minpts=10, maxpts=20):
        self.alpha = math.radians(alpha)
        self.beta = math.radians(beta)
        self.threshold_alpha = math.cos(self.alpha)
        self.threshold_beta = math.cos(self.beta)
        self.min_segments = min_segments
        self.minpts = minpts
        self.maxpts = maxpts

        self.pList = []
        self.clusters = []
        self.center_vectors = {}
        self.other_vectors = {}
        self.outlier = []
        self.fix_clusters = {}

        self.count = 0
        self.t_offline = 0

    def add(self, p, stop=None):
        # if (p is None) and (stop is True):
        # return self.result()
        self.pList.append(p)
        self.count += 1
        if self.count >= self.t_offline or stop:
            self.offline_cluster(stop)

        return self.clusters

    def check(self, segment: Segment):
        for cv, timeline in self.fix_clusters.items():
            if not timeline.crop(segment):
                continue
            segments, mapping_seg = timeline.crop(segment, returns_mapping=True)
            for seg in segments:
                if seg == segment:  # 重复
                    return False
                elif seg == mapping_seg[seg][-1]:
                    self.fix_clusters[cv] = timeline.remove(seg)
                    return True
                # if mapping_seg[seg][-1].duration <= segment.duration:  # 重叠
                #     self.fix_clusters[cv] = timeline.remove(mapping_seg[seg][-1])
                # return True  # 重叠 和 既不重复也不重叠的都需要执行下一步聚类

        # for p in self.pList:
        #     if p.segment & segment:
        #         if (p.segment & segment).duration > 1:
        #             p.segment = p.segment | segment
        #             return False

        return True

    def offline_cluster(self, stop):
        self.clusters = []

        if self.center_vectors:
            new_cvs, embeds = torch.Tensor(), torch.Tensor()
            for p in self.pList:
                embeds = torch.cat([embeds, p.embed], dim=0)
            for _, cv in self.center_vectors.items():
                new_cvs = torch.cat([new_cvs, cv], dim=0)
            sim_mx = self.cosine_similarity_matrix(new_cvs, embeds)
            to_delete, embeds = [], torch.Tensor()

            # for i in range(sim_mx.shape[0]):
            #     # 中心向量已经保证范围内有足够数量的数据点，直接对范围内的数据点聚成一类簇
            #     C = Cluster(name=f"c{len(self.clusters) + 1}")
            #     self.clusters.append(C)
            #     for j in range(len(self.pList)):
            #         if sim_mx[i][j] > self.threshold_alpha:
            #             C.add(self.pList[j].segment)
            #             to_delete += [j]
            #
            #     C.timeline = self.fix_clusters[f"CV{i + 1}"].union(C.timeline).support()
            #     self.fix_clusters[f"CV{i + 1}"] = C.timeline

            for i in range(sim_mx.shape[1]):
                # 中心向量已经保证范围内有足够数量的数据点，直接对范围内的数据点聚成一类簇
                # 对每个数据点找到范围内的最近的中心
                if max(sim_mx[:, i]) > self.threshold_beta:
                    idx = int(torch.argmax(sim_mx[:, i]))
                    self.fix_clusters[f"CV{idx + 1}"] = self.fix_clusters[f"CV{idx + 1}"].union(
                        Timeline([self.pList[i].segment])).support()
                    to_delete += [i]
            for i, timeline in enumerate(self.fix_clusters.values()):
                C = Cluster(name=f"c{i}")
                C.timeline = timeline
                self.clusters.append(C)

            for index in sorted(list(set(to_delete)), reverse=True):
                self.pList.remove(self.pList[index])

        # self.combine_cv()
        self.find_clusters2(stop)

    def offline_cluster2(self):
        self.clusters = []
        if self.center_vectors.numel():
            embeds = torch.Tensor()
            for p in self.pList:
                embeds = torch.cat([embeds, p.embed], dim=0)
            cos_matrix = self.cosine_similarity_matrix(self.center_vectors, embeds)

            # TODO: fuzzy border-----member_ship
            for i, cv in enumerate(self.center_vectors):
                indexes = torch.nonzero(cos_matrix[i] > self.threshold_alpha)
                indexes = [int(index[0]) for index in indexes]

                C = Cluster(name=f"c{len(self.clusters) + 1}")
                self.clusters.append(C)

                for index in indexes:
                    p = self.pList[index]

        self.find_clusters2()

    def find_clusters(self, embeds, not_visited):
        cos_matrix = self.cosine_similarity_matrix(embeds, embeds)
        for i in range(cos_matrix.shape[0]):
            indexes = torch.nonzero(cos_matrix[i] > self.threshold_cos)
            indexes = [int(index[0]) for index in indexes]
            indexes = [index for index in indexes if index in not_visited]

            if (len(indexes) > self.minpts) and (i in not_visited):
                C = Cluster(name=f"c{len(self.clusters) + 1}")
                self.clusters.append(C)
                for index in indexes:
                    # cos_matrix[:, index] = torch.zeros_like(cos_matrix[:, index])
                    C.add(self.pList[index].segment)
                    not_visited.remove(index)

                # 当某个类别中数据点的数量超过maxpts时，认为这个类别已经相当稳定了,计算中心向量代表类别
                if len(indexes) > self.maxpts:
                    center_vector = self.calculate_cv(indexes)
                    self.fix_clusters[f"CV{len(self.center_vectors) + 1}"] = C.timeline
                    self.delete_index += indexes
                    self.center_vectors = torch.cat([self.center_vectors, center_vector], dim=0)

        C = Cluster(name="outlier")
        self.clusters.append(C)
        for index in not_visited:
            C.add(self.pList[index].segment)

        for index in sorted(self.delete_index, reverse=True):
            self.pList.remove(self.pList[index])

    def find_clusters2(self, stop):
        embeds = torch.Tensor()
        for p in self.pList:
            embeds = torch.cat([embeds, p.embed], dim=0)

        if not embeds.numel():
            return

        sim_mx = self.cosine_similarity_matrix(embeds, embeds)
        threshold_matrix = sim_mx > self.threshold_alpha
        index = int(torch.argmax(sum(threshold_matrix)))

        candidate_embeds, candidate_delete, to_delete = torch.Tensor(), [], []
        not_visited = list(range(len(self.pList)))
        count = sum(threshold_matrix)[index]

        while count >= self.minpts:
            C = Cluster(f"c{len(self.clusters) + 1}")
            self.clusters.append(C)
            self.other_vectors[C.name] = self.pList[index].embed
            for i, similar in enumerate(threshold_matrix[index]):
                if similar:
                    C.add(self.pList[i].segment)
                    candidate_embeds = torch.cat([candidate_embeds, self.pList[i].embed], dim=0)
                    candidate_delete.append(i)
                    if i != index:
                        threshold_matrix[i, :] = False
                        threshold_matrix[:, i] = False

            threshold_matrix[index, :] = False
            threshold_matrix[:, index] = False
            if count >= self.maxpts:
                center_vector = torch.mean(candidate_embeds, dim=0, keepdim=True)
                self.fix_clusters[f"CV{len(self.center_vectors) + 1}"] = C.timeline
                self.center_vectors[f"cv{len(self.center_vectors) + 1}"] = center_vector

                to_delete += candidate_delete

            not_visited = [x for x in not_visited if x not in candidate_delete]
            candidate_embeds = torch.Tensor()
            candidate_delete = []
            index = int(torch.argmax(sum(threshold_matrix)))
            count = sum(threshold_matrix)[index]

        vectors, embeds = torch.Tensor(), torch.Tensor()
        for name, cv in self.center_vectors.items():
            vectors = torch.cat([vectors, cv], dim=0)
        for name, cv in self.other_vectors.items():
            vectors = torch.cat([vectors, cv], dim=0)
        self.other_vectors = {}
        if not vectors.numel():
            C = Cluster(name="outlier")
            self.clusters.append(C)
            for i in not_visited:
                C.add(self.pList[i].segment)
            return
        if not not_visited:
            return

        for i in not_visited:
            embeds = torch.cat([embeds, self.pList[i].embed], dim=0)
        sim_mx = self.cosine_similarity_matrix(vectors, embeds)
        max_idx = torch.max(sim_mx, dim=0).indices + 1

        for i, cluster in enumerate(self.clusters):
            for idx, j in enumerate(max_idx, start=0):
                if i == int(j):
                    cluster.add(self.pList[idx].segment)

        for i in sorted(to_delete, reverse=True):
            self.pList.remove(self.pList[i])

    def result(self):
        vectors, embeds = torch.Tensor(), torch.Tensor()
        for name, cv in self.center_vectors.items():
            vectors = torch.cat([vectors, cv], dim=0)
        for name, cv in self.other_vectors.items():
            vectors = torch.cat([vectors, cv], dim=0)

        if not vectors.numel():
            C = Cluster(name="c1")
            for i in self.not_visited:
                C.add(self.pList[i].segment)
            self.clusters.append(C)
            return self.clusters

        for i in self.not_visited:
            if len(self.pList) < i - 1:
                continue
            embeds = torch.cat([embeds, self.pList[i].embed], dim=0)
        sim_mx = self.cosine_similarity_matrix(vectors, embeds)
        max_idx = torch.max(sim_mx, dim=0).indices + 1

        for i, cluster in enumerate(self.clusters):
            for idx, j in enumerate(max_idx):
                if i == int(j):
                    cluster.add(self.pList[idx].segment)

        return self.clusters

    def update_notv(self, to_delete):
        del_list = sorted(to_delete)
        not_visited = sorted(self.not_visited)
        new_notv = []
        j = 0
        for i in not_visited:
            if del_list and del_list[j] < i:
                j += 1
            new_notv.append(i - j)
        self.not_visited = new_notv

    def calculate_cv(self, indexes):
        center_vector = self.pList[indexes[0]].embed
        size = len(indexes)
        for index in indexes[1:]:
            embed = self.pList[index].embed
            center_vector = center_vector + embed

        return center_vector / size

    def calculate_cv2(self, embeds):
        pass

    def combine_cv(self):
        """
        实验发现聚类结果通常存在超过7个的说话人情况，所以可能是存在一个说话人因为某些原因被分成了两个类别，
        因此期望对于相似度高的两个中心向量类进行结果合并，不改变保存在self.center_vectors中的信息。
        """
        for i in range(self.center_vectors.shape[0]):
            for j in range(self.center_vectors.shape[0]):
                pass

    def cosine_similarity_matrix(self, x, y):
        x = x / torch.linalg.norm(x, dim=1, keepdim=True)
        y = y / torch.linalg.norm(y, dim=1, keepdim=True)

        return torch.matmul(x, y.T)


class CosClustering2:
    def __init__(self, theta=55, alpha=60, min_segments=10, minpts=10, maxpts=20):
        self.theta = math.radians(theta)
        self.alpha = math.radians(alpha)
        self.threshold_theta = math.cos(self.theta)
        self.threshold_alpha = math.cos(self.alpha)
        self.min_segments = min_segments
        self.minpts = minpts
        self.maxpts = maxpts

        self.pList = []
        self.clusters = []
        self.center_vectors = {}
        self.other_vectors = {}
        self.outlier = []
        self.fix_clusters = {}

        self.count = 0
        self.t_offline = 20

    def add(self, p: DataPoint2, stop=None):
        self.pList.append(p)
        self.count += 1
        if self.count >= self.t_offline or stop:
            self.offline_cluster(stop)

        return self.clusters

    def offline_cluster(self, stop):
        # if len(self.pList) < self.min_segments:
        #     return
        self.clusters = []
        self.other_vectors = {}

        if self.center_vectors:
            new_cvs, embeds = torch.Tensor(), torch.Tensor()
            for p in self.pList:
                embeds = torch.cat([embeds, p.embed], dim=0)
            for _, detail in self.center_vectors.items():
                new_cvs = torch.cat([new_cvs, detail["embed"]], dim=0)
            sim_mx = self.cosine_similarity_matrix(new_cvs, embeds)
            to_delete, embeds = [], torch.Tensor()

            for i in range(sim_mx.shape[0]):
                # 中心向量已经保证范围内有足够数量的数据点，直接对范围内的数据点聚成一类簇
                C = Cluster2(name=f"c{len(self.clusters) + 1}")
                self.clusters.append(C)
                for j in range(len(self.pList)):
                    if sim_mx[i][j] > self.threshold_alpha:
                        embeds = torch.cat([embeds, self.pList[j].embed])
                        C.add(self.pList[j].index)
                        to_delete += [j]

                C.idxes = self.fix_clusters[f"CV{i + 1}"] + C.idxes
                self.fix_clusters[f"CV{i + 1}"] = C.idxes

            for index in sorted(list(set(to_delete)), reverse=True):
                self.pList.remove(self.pList[index])

        # self.combine_cv()
        self.find_clusters2(stop)

    def offline_cluster2(self):
        self.clusters = []
        if self.center_vectors.numel():
            embeds = torch.Tensor()
            for p in self.pList:
                embeds = torch.cat([embeds, p.embed], dim=0)
            cos_matrix = self.cosine_similarity_matrix(self.center_vectors, embeds)

            # TODO: fuzzy border-----member_ship
            for i, cv in enumerate(self.center_vectors):
                indexes = torch.nonzero(cos_matrix[i] > self.threshold_alpha)
                indexes = [int(index[0]) for index in indexes]

                C = Cluster(name=f"c{len(self.clusters) + 1}")
                self.clusters.append(C)

                for index in indexes:
                    p = self.pList[index]

        self.find_clusters2()

    def find_clusters(self, embeds, not_visited):
        cos_matrix = self.cosine_similarity_matrix(embeds, embeds)
        for i in range(cos_matrix.shape[0]):
            indexes = torch.nonzero(cos_matrix[i] > self.threshold_cos)
            indexes = [int(index[0]) for index in indexes]
            indexes = [index for index in indexes if index in not_visited]

            if (len(indexes) > self.minpts) and (i in not_visited):
                C = Cluster(name=f"c{len(self.clusters) + 1}")
                self.clusters.append(C)
                for index in indexes:
                    # cos_matrix[:, index] = torch.zeros_like(cos_matrix[:, index])
                    C.add(self.pList[index].segment)
                    not_visited.remove(index)

                # 当某个类别中数据点的数量超过maxpts时，认为这个类别已经相当稳定了,计算中心向量代表类别
                if len(indexes) > self.maxpts:
                    center_vector = self.calculate_cv(indexes)
                    self.fix_clusters[f"CV{len(self.center_vectors) + 1}"] = C.timeline
                    self.delete_index += indexes
                    self.center_vectors = torch.cat([self.center_vectors, center_vector], dim=0)

        C = Cluster(name="outlier")
        self.clusters.append(C)
        for index in not_visited:
            C.add(self.pList[index].segment)

        for index in sorted(self.delete_index, reverse=True):
            self.pList.remove(self.pList[index])

    def find_clusters2(self, stop):
        embeds = torch.Tensor()
        for p in self.pList:
            embeds = torch.cat([embeds, p.embed], dim=0)

        if not embeds.numel():
            return

        sim_mx = self.cosine_similarity_matrix(embeds, embeds)
        threshold_matrix = sim_mx > self.threshold_theta
        index = int(torch.argmax(sum(threshold_matrix)))

        candidate_embeds, candidate_delete, to_delete = torch.Tensor(), [], []
        not_visited = list(range(len(self.pList)))
        count = sum(threshold_matrix)[index]

        while count >= self.minpts:
            C = Cluster2(f"c{len(self.clusters) + 1}")
            self.clusters.append(C)
            self.other_vectors[C.name] = self.pList[index].embed
            for i, similar in enumerate(threshold_matrix[index]):
                if similar:
                    C.add(self.pList[i].index)
                    candidate_embeds = torch.cat([candidate_embeds, self.pList[i].embed], dim=0)
                    candidate_delete.append(i)
                    if i != index:
                        threshold_matrix[i, :] = False
                    threshold_matrix[:, i] = False

            threshold_matrix[index, :] = False
            threshold_matrix[:, index] = False
            if count >= self.maxpts:
                center_vector = torch.mean(candidate_embeds, dim=0, keepdim=True)
                self.fix_clusters[f"CV{len(self.center_vectors) + 1}"] = C.idxes
                self.center_vectors[f"cv{len(self.center_vectors) + 1}"] = {}
                self.center_vectors[f"cv{len(self.center_vectors)}"]["embed"] = center_vector
                self.center_vectors[f"cv{len(self.center_vectors)}"]["size"] = candidate_embeds.shape[0]

                to_delete += candidate_delete

            not_visited = [x for x in not_visited if x not in candidate_delete]
            candidate_embeds = torch.Tensor()
            candidate_delete = []
            index = int(torch.argmax(sum(threshold_matrix)))
            count = sum(threshold_matrix)[index]

        if stop:
            vectors, embeds = torch.Tensor(), torch.Tensor()
            for name, cv in self.center_vectors:
                vectors = torch.cat([vectors, cv], dim=0)
            for name, cv in self.other_vectors:
                vectors = torch.cat([vectors, cv], dim=0)
            for p in not_visited:
                embeds = torch.cat([embeds, p.embed], dim=0)
            sim_mx = self.cosine_similarity_matrix(vectors, embeds)
            max_idx = torch.max(sim_mx, dim=0).indices + 1

            for i, cluster in enumerate(self.clusters):
                for idx, j in enumerate(max_idx):
                    if i == int(j):
                        cluster.add(not_visited[idx].index)
        else:
            C = Cluster2(f"c{len(self.clusters) + 1}")
            self.clusters.append(C)
            for i in not_visited:
                # if now_timeline.crop(self.pList[i].segment):
                #     continue
                C.add(self.pList[i].index)
            for i in sorted(to_delete, reverse=True):
                self.pList.remove(self.pList[i])

    def calculate_cv(self, indexes):
        center_vector = self.pList[indexes[0]].embed
        for index in indexes[1:]:
            embed = self.pList[index].embed
            center_vector = (center_vector + embed) / 2

        return center_vector

    def calculate_cv2(self, embeds):
        pass

    def combine_cv(self):
        """
        实验发现聚类结果通常存在超过7个的说话人情况，所以可能是存在一个说话人因为某些原因被分成了两个类别，
        因此期望对于相似度高的两个中心向量类进行结果合并，不改变保存在self.center_vectors中的信息。
        """
        for i in range(self.center_vectors.shape[0]):
            for j in range(self.center_vectors.shape[0]):
                pass

    def cosine_similarity_matrix(self, x, y, eps=1e-15):
        # 计算向量的L2范数
        x = x / torch.linalg.norm(x, dim=-1, keepdims=True).clip(min=eps)
        y = y / torch.linalg.norm(y, dim=-1, keepdims=True).clip(min=eps)

        return torch.matmul(x, y.T)

    def cosine(self, x, y, eps=1e-15):
        x = x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps)
        y = y / np.linalg.norm(y, axis=-1, keepdims=True).clip(min=eps)
        return np.dot(x, y.T)
