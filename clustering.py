from LinearAlgebra import Vector, vector_mean, squared_distance, distance
from typing import List, NamedTuple, Union, Callable, Tuple
import itertools, random, tqdm
import matplotlib.image as mpimg
from matplotlib import pyplot as plt


def num_differences(v1: Vector, v2: Vector) -> int:
    assert len(v1) == len(v2)
    return len([x1 for x1, x2 in zip(v1, v2) if x1 != x2])


assert num_differences([1, 2, 3], [2, 1, 3]) == 2
assert num_differences([1, 2], [1, 2]) == 0


def cluster_means(k: int,
                  inputs: List[Vector],
                  assignments: List[int]) -> List[Vector]:
    # cluster[i] содержит входы, чье значение равно i
    clusters = [[] for i in range(k)]
    for input, assignment in zip(inputs, assignments):
        clusters[assignment].append(input)

    # Если кластер пустой, то взять случайную точку
    return [vector_mean(cluster) if cluster else random.choice(inputs)
            for cluster in clusters]


class KMeans:
    def __init__(self, k: int) -> None:
        self.k = k  # число кластеров
        self.means = None

    def classify(self, input: Vector) -> int:
        """вернуть индекс кластера, ближайшего к входному значению"""
        return min(range(self.k),
                   key=lambda i: squared_distance(input, self.means[i]))

    def train(self, inputs: List[Vector]) -> None:
        # начать со случайных значений
        assignments = [random.randrange(self.k) for _ in inputs]
        with tqdm.trange(itertools.count()) as t:
            for _ in t:
                # вычислить среднее и отыскать новые значения
                self.means = cluster_means(self.k, inputs, assignments)
                new_assignments = [self.classify(input)
                                   for input in inputs]

                # проверить, сколько назначений изменилось,
                # и если несколько, то работа завершена
                num_changed = num_differences(assignments, new_assignments)

                if num_changed == 0:
                    return

                # в противном случае оставить новые значения и
                # вычислить новые средние
                assignments = new_assignments
                self.means = cluster_means(self.k, inputs, assignments)
                t.set_description(f"changed:{num_changed} / {len(inputs)}")


def squared_clustering_errors(inputs: List[Vector], k: int) -> float:
    """отыскивает сумму квадратов ошибок, возникающих
    из кластеризации входов k средними"""
    clusterer = KMeans(k)
    clusterer.train(inputs)
    means = clusterer.means
    assignments = [clusterer.classify(input) for input in inputs]

    return sum(squared_distance(input, means[cluster])
               for input, cluster in zip(inputs, assignments))


# ____ВОСХОДЯЩАЯ ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ______

class Leaf(NamedTuple):
    value: Vector


leaf1 = Leaf([10, 20])
leaf2 = Leaf([30, -15])


class Merged(NamedTuple):
    children: tuple
    order: int


merged = Merged((leaf1, leaf2), order=1)

Cluster = Union[Leaf, Merged]


def get_values(cluster: Cluster) -> List[Vector]:
    if isinstance(cluster, Leaf):
        return [cluster.value]
    else:
        return [value
                for child in cluster.children
                for value in get_values(child)]


assert get_values(merged) == [[10, 20], [30, -15]]


def cluster_distance(cluster1: Cluster,
                     cluster2: Cluster,
                     distance_agg: Callable = min) -> float:
    """вычислить все попарные расстояния между cluster1 и cluster2
    и применить агрегатную функцию _distance_agg_
    к результирующему списку"""
    return distance_agg([distance(v1, v2)
                         for v1 in get_values(cluster1)
                         for v2 in get_values(cluster2)])


def get_merge_order(cluster: Cluster) -> float:
    if isinstance(cluster, Leaf):
        return float('inf')  # ни разу не объединялся
    else:
        return cluster.order


def get_children(cluster: Cluster):
    if isinstance(cluster, Leaf):
        raise TypeError(" Лист не имеет дочерних элементов")
    else:
        return cluster.children


# создание кластерного алгоритма
def bottom_up_cluster(inputs: List[Vector],
                      distance_agg: Callable = min) -> Cluster:
    # начать с того, что все элементы являются листьями
    clusters: List[Cluster] = [Leaf(input) for input in inputs]

    def pair_distance(pair: Tuple[Cluster, Cluster]) -> float:
        return cluster_distance(pair[0], pair[1], distance_agg)

    # до тез пор пока есть более одного кластера
    while len(clusters) > 1:
        # отыскать два ближайших кластера
        c1, c2 = min(((cluster1, cluster2)
                      for i, cluster1 in enumerate(clusters)
                      for cluster2 in clusters[:i]),
                     key=pair_distance)
        # удалить их из списка кластеров
        clusters = [c for c in clusters if c != c1 and c != c2]

        # объеденить их, тспользуя merge_order = число оставшихся кластеров
        merged_cluster = Merged((c1, c2), order=len(clusters))

        # и добавить их объединение
        clusters.append(merged_cluster)

    # когда остался всего один кластер, вернуть его
    return clusters[0]


inputs = [[-14, -5], [13, 13], [20, 23], [-19, -11], [-9, -16], [21, 27], [-49, 15], [26, 13], [-46, 5], [-34, -1],
          [11, 15], [-49, 0], [-22, -16], [19, 28], [-12, -8], [-13, -19], [-41, 8], [-11, -6], [-25, -9], [-18, -3]]

base_cluster = bottom_up_cluster(inputs)


def generate_clusters(base_cluster: Cluster,
                      num_clusters: int) -> List[Cluster]:
    # начать со списка, состоящего только из базового кластера
    clusters = [base_cluster]

    # до тех пор, пока кластеров недостаточно
    while len(clusters) < num_clusters:
        # выбрать из кластеров тот, который объединен с последним
        next_cluster = min(clusters, key=get_merge_order)

        # удалить его из списка
        clusters = [c for c in clusters if c != next_cluster]

        # добавить его дочерние элементы в список (разделить его)
        clusters.extend(get_children(next_cluster))

    # как только достаточно кластеров
    return clusters


# генерация трех кластеров
three_clusters = [get_values(cluster)
                  for cluster in generate_clusters(base_cluster, 3)]

# вывод их на график
for i, cluster, marker, color in zip([1, 2, 3],
                                    three_clusters,
                                    ['D', 'o', '*'],
                                    ['r', 'g', 'b']):
    xs, ys = zip(*cluster)    # распаковка
    plt.scatter(xs, ys, color = color, marker = marker)

    # установить число на среднем значении кластера
    x, y = vector_mean(cluster)
    plt.plot(x, y, marker = '$' + str(i) + '$', color = 'black')
    plt.title(" Места проживания - 3 кластера снизу вверх, min")
    plt.xlabel("Кварталы к востоку от центра города")
    plt.ylabel("Кварталы к северу от центра города")
    plt.show()
