from typing import NamedTuple, Dict, List, Tuple
from collections import deque, Counter
from LinearAlgebra import Matrix, make_matrix, shape, Vector, dot, magnitude, distance
import random, tqdm

class User(NamedTuple):
    id: int
    name: str


users = [
    {"id": 0, "name": "Hero"},
    {"id": 1, "name": "Dunn"},
    {"id": 2, "name": "Sue"},
    {"id": 3, "name": "Chi"},
    {"id": 4, "name": "Thor"},
    {"id": 5, "name": "Clive"},
    {"id": 6, "name": "Hicks"},
    {"id": 7, "name": "Devin"},
    {"id": 8, "name": "Kate"},
    {"id": 9, "name": "Klein"}
]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]


# псевдонимы типов для отслеживания дружеских отношений

Friendships = Dict[int, List[int]]

friendships: Friendships = {user.id: [] for user in users}


for i, j in friend_pairs:
    friendships[i].append(j)
    friendships[j].append(i)

# ________ЦЕНТРАЛЬНОСТЬ ПО СПОСРЕДНИЧЕСТВУ___________

Path = List[int]


def shortest_paths_from(from_user_id: int,
                        friendships: Friendships) -> Dict[int, List[Path]]:
    # словарь из user_id во *все* кратчайшие пути до этого пользователя
    shortest_paths_to: Dict[int, List[Path]] = {from_user_id: [[]]}

    # Очередь (предыдущий пользователь, следующий пользователь),
    # которую необходимо проверить, начинается со всех пар
    # (from_user, friend_of_from_user), т.е. пользователя и его друга
    frontier = deque((from_user_id, friend_id)
                     for friend_id in friendships[from_user_id])

    # продолжать до тех пор, пока очередь не станет пустой
    while frontier:
        # удалить пару, которая является следующей в очереди
        prev_user_id, user_id = frontier.popleft()

        # из-за способа добавления в очередь, некоторые кратчайшие пути до prev_user уже известны
        paths_to_prev_user = shortest_paths_to[prev_user_id]
        new_paths_to_user = [path + [user_id]
                             for path in paths_to_prev_user]

        # возможно кратчайший путь до user_id уже известен
        old_path_to_user = shortest_paths_to.get(user_id, [])

        # поиск кратчайшего и ранее неизвестного пути
        if old_path_to_user:
            min_path_length = len(old_path_to_user[0])
        else:
            min_path_length = float('inf')

        # оставить только не слишком длинные
        # и фактически новые пути
        new_paths_to_user = [path
                             for path in new_paths_to_user
                             if len(path) <= min_path_length
                             and path not in old_path_to_user]

        shortest_paths_to[user_id] = old_path_to_user + new_paths_to_user

        # добавить неизвестных соседей frontier
        frontier.extend((user_id, friend_id)
                        for friend_id in friendships[user_id]
                        if friend_id not in shortest_paths_to)

    return shortest_paths_to


# для каждого from_user и для каждого to_user список кратчайших путей
shortest_paths = {user.id: shortest_paths_from(user.id, friendships)
                  for user in users}

betweenness_centrality = {user.id: 0.0 for user in users}

for source in users:
    for target_id, paths in shortest_paths[source.id].items():
        if source.id < target_id:  # не дублировать подсчет
            num_paths = len(paths)  # кол-во кратчайших путей
            contrib = 1 / num_paths  # вклад в центральность
            for path in paths:
                for between_id in path:
                    if between_id not in [source.id, target_id]:
                        betweenness_centrality[between_id] += contrib


# __________ЦЕНТРАЛЬНОСТЬ ПО БЛИЗОСТИ_________
def farness(user):
    """сумма длин кратчейших путей до каждого пользователя"""
    return sum(len(paths[0])
               for paths in user["shortest_paths"].value())


closeness_centrality = {user.id: 1 / farness(user.id) for user in users}


# ___________ЦЕНТРАЛЬНОСТЬ ПО СОБСТВЕННОМУ ВЕКТОРУ__________

def matrix_times_matrix(m1: Matrix, m2: Matrix) -> Matrix:
    nr1, nc1 = shape(m1)
    nr2, nc2 = shape(m2)

    assert nc1 == nc2, "(число столбцов в m1 == числу строк в m2)"

    def entry_fn(i: int, j: int) -> float:
        """скалярное произведение
        i-й строки матрицы m1 и j-го столбца матрицы m2"""
        return sum(m1[i][k] * m2[k][j]
                   for k in range(nc1))

    return make_matrix(nr1, nc2, entry_fn)


def matrix_times_vector(m: Matrix, v: Vector) -> Vector:
    nr, nc = shape(m)
    n = len(v)
    assert nc == n, "(число столбцов m) == (число элементов v)"

    return [dot(row, v) for row in m]


def find_eigenvector(m: Matrix,
                     tolerance: float = 0.00001) -> Tuple[Vector, float]:
    guess = [random.random() for _ in m]

    while True:
        result = matrix_times_vector(m, guess)   # преобразовать догадку
        norm = magnitude(result)    # вычислить норму
        next_guess = [x / norm for x in result]    # перешкалировать

        if distance(guess, next_guess) < tolerance:
            # схождение, вернуть
            # (собственый векор, собственное значение)
            return next_guess, norm

        guess = next_guess

#______ЦЕНТРАЛЬНОСТЬ______
def entry_fn(i: int, j: int):
    return 1 if (i, j) in friend_pairs or (j, i) in friend_pairs else 0

n = len(users)
adjacency_matrix = make_matrix(n, n, entry_fn)

eigenvector_centralities, _ = find_eigenvector(adjacency_matrix)



#________ОРИЕНТИРОВАННЫЕ ГРАФЫ И АЛГОРИТМ PageRank_______

endorsements = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1), (1, 3),
                (2, 3), (3, 4), (5, 4), (5, 6), (7, 5), (6, 8), (8, 7), (8, 9)]

endorsement_counts = Counter(target for source, target in endorsements)

        
def page_rank(users: List[User],
              endoresements: List[Tuple[int, int]],
              damping: float = 0.85,
              num_iters: int = 100) -> Dict[int, float]:
    # вычислить, сколько людей пользователь
    #  поддерживает своим авторитетом
    outgoing_counts = Counter( target for source, target in endoresements)

    # первоначально распределить ранг PageRank равномерно
    num_users = len(users)
    pr = {user.id: 1 / num_users for user in users}

    # малая доля ранга PageRank, которую каждый узел получает
    # на каждой итерации
    base_pr = ( 1 - damping) / num_users

    for iter in tqdm.trange(num_iters):
        next_pr = {user.id: base_pr for user in users}    # начать с base_pr

        for source, target in endoresements:
            # добавить демпфированную долю
            # исходного рейтинга source в цель target
            next_pr[target] += damping * pr[source] / outgoing_counts[source]

        pr = next_pr

    return pr


pr = page_rank(users, endorsements)

assert pr[4] > max(page_rank
                   for user_id, page_rank in pr.items()
                   if user_id != 4)
