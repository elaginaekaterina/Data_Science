from typing import List, Iterator, Tuple, Iterable, Callable, Any, NamedTuple
from collections import Counter, defaultdict
import math, random, re, datetime


def tokenize(document: str) -> List[str]:
    """просто разбить по пробелу"""
    return document.split()


def word_count_old(documents: List[str]):
    """подсчет количества появлений слов
    без использования алгоритма MapReduce"""
    return Counter(word
                   for document in documents
                   for word in tokenize(document))


def wc_mapper(document: str) -> Iterator[Tuple[str, int]]:
    """для каждого слова в документе эмитировать (слово, 1)"""
    for word in tokenize(document):
        yield (word, 1)


def wc_reducer(word: str,
               counts: Iterable[int]) -> Iterator[Tuple[str, int]]:
    """просуммировать количества появлений для слова"""
    yield (word, sum(counts))


def word_count(documents: List[str]) -> List[Tuple[str, int]]:
    """подсчитывает количества появлений слов в выходных документах,
    используя алгоритм MapReduce"""
    collector = defaultdict(list)  # сохранить группированные значения

    for document in documents:
        for word, count in wc_mapper(document):
            collector[word].append(count)

    return [output
            for word, counts in collector.items()
            for output in wc_reducer(word, counts)]


# MapReduce в более общем плане
# пара ключ/значение - просто 2-ленный кортеж
KV = Tuple[Any, Any]

# преобразователь - ф-ия, возвращающая итерируемый объект
# Iterable, состоящий из пар клю/значение
Mapper = Callable[..., Iterable[KV]]

# редуктор - ф-я, которая берет ключ и итерируемый объект со значениями
# и возвращает пару ключ/значение
Reducer = Callable[[Any, Iterable], KV]


def map_reduce(inputs: Iterable,
               mapper: Mapper,
               reducer: Reducer) -> List[KV]:
    """пропустить выходы через MapReduce,
    используя преобразователь и редуктор"""
    collector = defaultdict(list)

    for input in inputs:
        for key, value in mapper(input):
            collector[key].append(value)

    return [output
            for key, values in collector.items()
            for output in reducer(key, values)]


word_counts = map_reduce(documents, wc_mapper, wc_reducer())


def values_reducer(values_fn: Callable) -> Reducer:
    """вернуть редуктор, который просто применяет функцию
    values_fn к своим значениям"""

    def reduce(key, values: Iterable) -> KV:
        return (key, values_fn(values))

    return reduce


sum_reducer = values_reducer(sum)
max_reducer = values_reducer(max)
min_reducer = values_reducer(min)
count_distinct_reducer = values_reducer(lambda values: len(set(values)))

# пример "анализ обновлений новостной ленты"

status_updates = [
    {"id": 1,
     "username": "joelgrus",
     "text": "Is anyone interested in a data science book?",
     "created_at": datetime.datetime(2018, 12, 21, 11, 47, 0),
     "liked_by": ["data_guy", "data_gal", "bill"]},
    {"id": 2,
     "username": "ekaterinaelagina",
     "text": " I should learn a data science.",
     "create_at": datetime.datetime(2022, 9, 5, 12, 21, 5),
     "liked_by": ["data_guy", "data_gal", "bill"]},
]


def data_science_day_mapper(status_update: dict) -> Iterable:
    """выдает (день_недели, 1), если обновление ленты
    новостей содержит "data science" """
    if "data science" in status_update["text"].lower():
        day_of_week = status_update["created_at"].weekday()
        yield (day_of_week, 1)


data_science_days = map_reduce(status_updates,
                               data_science_day_mapper,
                               sum_reducer)


def words_per_user_mapper(status_update: dict):
    user = status_update["username"]
    for word in tokenize(status_update["text"]):
        yield (user, (word, 1))


def most_popular_word_reducer(user: str,
                              words_and_counts: Iterable[KV]):
    """с учетом последовательности из пар (слово, количество)
    вернуть слово наивысшим суммарным количеством появлений"""
    word_counts = Counter()
    for word, count in words_and_counts:
        word_counts[word] += count
    word, count = word_counts.most_common(1)[0]

    yield (user, (word, count))


user_words = map_reduce(status_updates,
                        words_per_user_mapper,
                        most_popular_word_reducer)


# преобразователь поклонников
def liker_mapper(status_update: dict):
    user = status_update["username"]
    for liker in status_update["liked_by"]:
        yield (user, liker)


distinct_likers_per_user = map_reduce(status_updates,
                                      liker_mapper,
                                      count_distinct_reducer)


# Пример "умножение матриц"

# элементы матрицы
class Entry(NamedTuple):
    name: str
    i: int
    j: int
    value: float


def matrix_multiply_mapper(num_rows_a: int, num_cols_b: int) -> Mapper:
    """m - это общий размер (столбцы A, строки B)
        элемент представляет собой кортеж (matrix_name, i, j, значение)"""

    def mapper(entry: Entry):
        if entry.name == "A":
            for y in range(num_cols_b):
                key = (entry.i, y)  # какой элемент C
                value = (entry.j, entry.value)  # какой элемент в сумме
                yield (key, value)
        else:
            for x in range(num_rows_a):
                key = (x, entry.j)
                value = (entry.i, entry.value)
                yield (key, value)

    return mapper


def matrix_multiply_reducer(key: Tuple[int, int],
                            indexed_values: Iterable[Tuple[int, int]]):
    results_by_index = defaultdict(list)

    for index, value in indexed_values:
        results_by_index[index].append(value)

    # умножить значения для позиций с двумя знаниями
    # (одно из A и одно из B) и суммировать их
    sumproduct = sum(values[0] * values[1]
                     for values in results_by_index.values()
                     if len(values) == 2)

    if sumproduct != 0.0:
        yield (key, sumproduct)

entries = [("A", 0, 0, 3), ("A", 0, 1,  2),
           ("B", 0, 0, 4), ("B", 0, 1, -1), ("B", 1, 0, 10)]
mapper = matrix_multiply_mapper(num_rows_a = 2, num_cols_b = 3)
reducer = matrix_multiply_reducer
