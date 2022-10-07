import random
from typing import TypeVar, List, Tuple

X = TypeVar('X')  # обобщенный тип для представления точки данных


def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """разбить данные на доли [prob, 1 - prob]"""
    data = data[:]  # Сделать мелкую копию,
    random.shuffle(data)  # т.к shuffle модифицирует список.
    cut = int(len(data) * prob)  # Применить prob для отыскания отсечпения
    return data[:cut], data[cut:]  # и рзбить там перетасованный список


data = [n for n in range(1000)]
train, test = split_data(data, 0.75)

assert len(train) == 750
assert len(test) == 250

assert sorted(train + test) == data

# если входные переменные спарены
Y = TypeVar('Y')  # обобщенный тип для представления входных переменных


def train_test_split(xs: List[X],
                     ys: List[Y],
                     test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    # генерация индексов и их разбивка
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],
            [xs[i] for i in test_idxs],
            [ys[i] for i in train_idxs],
            [ys[i] for i in test_idxs])


xs = [x for x in range(1000)]
ys = [2 * x for x in xs]
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test) == 250

# проверить, что соответствующие точки данных спарены правильно
assert all(y == 2 * x for x, y in zip(x_train, y_train))
assert all(y == 2 * x for x, y in zip(x_test, y_test))


# model = SomeKindOfModel()
# x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.33)
# model.train(x_train, y_train)
# perfomance = model.test(x_test, y_test)


# ПРАВИЛЬНОСТЬ, ТОЧНОСТЬ И ПРЕЦИЗИОННОСТЬ

def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total


assert accuracy(70, 4930, 13930, 981070) == 0.98114


def precision(tp: int, fp: int, tn: int) -> float:
    return tp / (tp + fp)


assert precision(70, 4930, 13930) == 0.014


def recall(tp: int, fp: int, fn: int, tn: int) -> float:
    return tp / (tp + fn)


assert recall(70, 4930, 13930, 981070) == 0.005


def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)

    return 2 * p * r / (p + r)
