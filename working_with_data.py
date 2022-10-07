import datetime
from typing import List, Dict
from collections import Counter
import math
import matplotlib.pyplot as plt
import random

import tqdm

from Probability import inverse_normal_cdf


# ОДНОМЕРНЫЕ ДАННЫЕ
def bucketize(point: float, bucket_size: float) -> float:
    """Округлить точку до следующего наименьшего кратного
    размера интервала bucket_size"""
    return bucket_size * math.floor(point / bucket_size)


def make_hiatogram(points: List[float], bucket_size: float) -> Dict[float, int]:
    """Разбивает точки на интервалы и подсчитывает
    их количество в каждом интервале"""
    return Counter(bucketize(point, bucket_size) for point in points)


def plot_histogram(points: List[float], bucket_size: float,
                   title: str = ""):
    histogram = make_hiatogram(points, bucket_size)
    plt.bar(list(histogram.keys()),
            list(histogram.values()), width=bucket_size)
    plt.title(title)
    plt.show()


# рассмотрим следующие 2 набора данных
random.seed(0)

# равномерное распределение между -100 и 100
uniform = [200 * random.random() - 100 for _ in range(10000)]

# нормельное распределение со средним 0, стандартным отклонением 57
normal = [57 * inverse_normal_cdf(random.random())
          for _ in range(10000)]

plot_histogram(uniform, 10, "Равномерная гистограмма")

plot_histogram(normal, 10, "Нормальная гистограмма")


# ДВУМЕРНЫЕ ДАННЫЕ

def random_normal():
    """Возвращает случайную выборку из стандартного
    нормального распределения"""
    return inverse_normal_cdf(random.random())


xs = [random_normal() for _ in range(1000)]
ys1 = [x + random_normal() / 2 for x in xs]
ys2 = [-x + random_normal() / 2 for x in xs]

plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title("Совсем разные совместные распределения")
plt.show()

from Statistika import correlation

# print(correlation(xs,ys1))
# print(correlation(xs, ys2))

# МНОГОЧИСЛЕННЫЕ РАЗМЕРНОСТИ
from LinearAlgebra import Matrix, Vector, make_matrix


# 1. матрица корреляций
def correlation_matrix(data: List[Vector]) -> Matrix:
    """
    Возвращает матрицк размера len(data) x len(data),
    (i, j)-й элемент которой является корреляцией между data[i] и data[j]
    """

    def correlation_ij(i: int, j: int) -> float:
        return correlation(data[i], data[j])

    return make_matrix(len(data), len(data), correlation_ij)


# 2. матрица рассеяний
num_points = 100


def random_row() -> List[float]:
    row = [0.0, 0, 0, 0]
    row[0] = random_normal()
    row[1] = -5 * row[0] + random_normal()
    row[2] = row[0] + row[1] + 5 * random_normal()
    row[3] = 6 if row[2] > -2 else 0
    return row


random.seed(0)
# в каждой строке по 4 точки
corr_rows = [random_row() for _ in range(num_points)]

corr_data = [list(col) for col in zip(*corr_rows)]

# данные corr_data - это список из четырех 100-мерных векторов
num_vectors = len(corr_data)
fig, ax = plt.subplots(num_vectors, num_vectors)

for i in range(num_vectors):
    for j in range(num_vectors):
        # разбросать столбец j по оси x напротив столбца i на оси y
        if i != j:
            ax[i][j].scatter(corr_data[j], corr_data[i])
        # Если не i == j, то в этом случае показать имя серии
        else:
            ax[i][j].annotate("Серия " + str(i), (0.5, 0.5),
                              xycoords='axes fraction',
                              ha="center", va="center")
        # затем спрятать осевые метки, за исключением
        # левой и нижней диаграмм
        if i < num_vectors - 1: ax[i][j].xaxis.set_visible(False)
        if j > 0: ax[i][j].yaxis.set_visible(False)

# Настроить правую нижнюю и левую верхнюю осевые метки,
# которые некорректны потому что на их диаграммах выводится только текст
ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
ax[0][0].set_ylim(ax[0][1].get_ylim())
plt.show()

# ТИПИЗИРОВАННЫЕ ИМЕНОВАННЫЕ КОРТЕЖИ
from typing import NamedTuple


class StockPrice(NamedTuple):
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        """Это класс, и поэтому мы можем добавлять методы"""
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']


price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.03
assert price.is_high_tech()

# КЛАССЫ ДАННЫХ dataclasses

from dataclasses import dataclass


@dataclass
class StockPrice2:
    symbol: str
    date: datetime.date
    closing_price: float

    def is_high_tech(self) -> bool:
        """Это класс, и поэтому мы можем добавлять методы"""
        return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']


price2 = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price2.symbol == 'MSFT'
assert price2.closing_price == 106.03
assert price2.is_high_tech()

# _____ОЧИСТКА И КОНВЕРТИРОВАНИЕ_____
from dateutil.parser import parse


def parse_row(row: List[str]) -> StockPrice:
    symbol, date, closing_price = row
    return StockPrice(symbol=symbol,
                      date=parse(date).date(),
                      closing_price=float(closing_price))


# тестирование функции
stock = parse_row(["MSFT", "2018-12-14", "106.03"])

assert stock.symbol == "MSFT"
assert stock.date == datetime.date(2018, 12, 14)
assert stock.closing_price == 106.03

# в случае плохих данных предпочтительнее получить None, чем сбой программы
from typing import Optional
import re


def try_parse_row(row: List[str]) -> Optional[StockPrice]:
    symbol, date, closing_price = row
    # символ акции должет состоять только из прописных букв
    if not re.match(r"^[A-Z]+$", symbol):
        return None

    try:
        date = parse(date).date()
    except ValueError:
        return None

    try:
        closing_price = float(closing_price)
    except ValueError:
        return None

    return StockPrice(symbol, date, closing_price)


assert try_parse_row(["MSFT0", "2018-12-14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12--14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12-14", "x"]) is None

assert try_parse_row(["MSFT", "2018-12-14", "106.03"]) == stock

# если данные разлелены запятыми и имеют плохие данные
import csv

data: List[StockPrice] = []

with open("comma_delimited_stock_prices .csv") as f:
    reader = csv.reader(f)
    for row in reader:
        maybe_stock = try_parse_row(row)
        if maybe_stock is None:
            print(f"Пропуск недопустимой строки данных:{row}")
        else:
            data.append(maybe_stock)

# ______ОПЕРИРОВАНИЕ ДАННЫМИ_____


from collections import defaultdict

max_prices: Dict[str, float] = defaultdict(lambda: float('-inf'))

for sp in data:
    symbol, closing_price = sp.symbol, sp.closing_price
    if closing_price > max_prices[symbol]:
        max_prices[symbol] = closing_price

# вычисление процентного изменения
from typing import List
from collections import defaultdict

# собрать цены по символу
prices: Dict[str, List[StockPrice]] = defaultdict(list)

for sp in data:
    prices[sp.symbol].append(sp)

# упорядочить цены по дате
prices = {symbol: sorted(symbol_prices)
          for symbol, symbol_prices in prices.items()}


def pct_change(yesterday: StockPrice, today: StockPrice) -> float:
    return today.closing_price / yesterday.closing_price - 1


class DailyChange(NamedTuple):
    symbol: str
    data: datetime.date
    pct_change: float


def day_over_day_changes(prices: List[StockPrice]) -> List[DailyChange]:
    """ предполагает, что цены только для одной акции и упорядочены"""
    return [DailyChange(symbol=today.symbol,
                        date=today.date,
                        pct_change=pct_change(yesterday, today))
            for yesterday, today in zip(prices, prices[1:])]


all_changes = [change
               for symbol_prices in prices.values()
               for change in day_over_day_changes(symbol_prices)]

max_change = max(all_changes, key=lambda change: change.pct_change)

min_change = min(all_changes, key=lambda change: change.pct_change)

changes_by_month: List[DailyChange] = {month: [] for month in range(1, 13)}

for change in all_changes:
    changes_by_month[change.data.month].append(change)

avg_daily_change = {
    month: sum(change.pct_change for change in changes) / len(changes)
    for month, changes in changes_by_month.items()
}

# _______ШКАЛИРОВАНИЕ______

from typing import Tuple
from LinearAlgebra import vector_mean
from Statistika import standard_deviation


def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
    """ возвращает среднее значение и стандартное отклонение
    для каждой позиции"""
    dim = len(data[0])

    means = vector_mean(data)

    stdevs = [standard_deviation([vector[i] for vector in data])
              for i in range(dim)]

    return means, stdevs


vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
means, stdevs = scale(vectors)


def rescale(data: List[Vector]) -> List[Vector]:
    """шкалирует входные данные так, чтобы каждый столбец
    имел нулевое среднее значение и стандартное отклонение, равное 1
    (оставляет позицию как есть, если ее стандартноеотклонение равно 0)"""
    dim = len(data[0])
    means, stdevs = scale(data)

    # сделать копию каждого вектора
    rescaled = [v[:] for v in data]
    for v in rescaled:
        for i in range(dim):
            if stdevs[i] > 0:
                v[i] = (v[i] - means[i]) / stdevs[i]
    return rescaled


means, stdevs = scale(rescale(vectors))

# СНИЖЕНИЕ РАЗМЕРНОСТИ

from LinearAlgebra import subtract, magnitude, dot, scalar_multiply


def de_mean(data: List[Vector]) -> List[Vector]:
    """Процентрировать данные, чтобы иметь среднее,
    равное 0, в каждой размерности"""
    mean = vector_mean(data)
    return [subtract(vector, mean) for vector in data]


def direction(w: Vector) -> Vector:
    mag = magnitude(w)
    return [w_i / mag for w_i in w]


def direction_variance(data: List[Vector], w: Vector) -> float:
    """Возвращает дисперсию x в направлении w"""
    w_dir = direction(w)
    return sum(dot(v, w_dir) ** 2 for v in data)


def direction_variance_gradient(data: List[Vector], w: Vector) -> Vector:
    """Градиент направленной дисперсии по отношению к w"""
    w_dir = direction(w)
    return [sum(2 * dot(v, w_dir) * v[i] for v in data)
            for i in range(len(w))]


from gradient_descent import gradient_step


def first_principal_component(data: List[Vector],
                              n: int = 100,
                              step_size: float = 0.1) -> Vector:
    # начать со случайной догадки
    guess = [1.0 for _ in data[0]]
    with tqdm.trange(n) as t:
        for _ in t:
            dv = direction_variance(data, guess)
            gradient = gradient_step(guess, gradient, step_size)
            t.set_description(f"dv: {dv:.3f}")

    return direction(guess)


def project(v: Vector, w: Vector) -> Vector:
    """вернуть проекуцию v на направление w"""
    projection_legth = dot(v, w)
    return scalar_multiply(projection_legth, w)


def remove_projection_from_vector(v: Vector, w: Vector) -> Vector:
    """проецирует v на w вычитает результат из v"""
    return subtract(v, project((v, w)))


def remove_projection(data: List[Vector], w: Vector) -> List[Vector]:
    return [remove_projection_from_vector(v, w) for v in data]


# анализ главных компонент
def pca(data: List[Vector], num_components: int) -> List[Vector]:
    components: List[Vector] = []
    for _ in range(num_components):
        component = first_principal_component(data)
        components.append(component)
        data = remove_projection(data, component)

    return components


def transform_vector(v: Vector, components: List[Vector]) -> Vector:
    return [dot(v, w) for w in components]


def transform(data: List[Vector], components: List[Vector]) -> List[Vector]:
    return [transform_vector(v, components) for v in data]
