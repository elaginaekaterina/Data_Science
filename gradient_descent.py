from typing import Callable, TypeVar, List, Iterator
import matplotlib.pyplot as plt
import random
from LinearAlgebra import Vector, dot, distance, add,scalar_multiply, vector_mean

def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h

def square(x: float) -> float:
    return x ** x

def derivative(x: float) -> float:
    return 2 * x

xs = range(-10, 11)
actuals = [derivative(x) for x in xs]
estimates = [difference_quotient(square, x, h = 0.001) for x in xs]
# построение графика
'''
plt.title("Фактические производные и их оценки")
plt.plot(xs, actuals, 'rx', label = 'Факт') # красный x
plt.plot(xs, estimates, 'b+', label = 'Оценка') # синий +
plt.legend(loc = 9)
plt.show()'''


# Частное разностное отношение
def partitial_difference_quotient(f: Callable[[Vector], float],
                                  v: Vector,
                                  i: int,
                                  h: float) -> float:
    """ возвращает i-e частное разностное отношение функции f в v"""
    w = [v_j + (h if j == i else 0) # добавить h только в i-й элемент v
         for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

# вычисление градиента
def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector, h: float = 0.0001):
    return [partitial_difference_quotient(f, v, i, h)
            for i in range(len(v))]

# Использование градиента
def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """ движется с шагом 'step_size' в направлении
    градиента 'gradient' от 'v' """
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]

# подбор случайной отправной точки
v = [random.uniform(-10, 10) for i in range(3)]

for epoch in range(1000):
    grad = sum_of_squares_gradient(v) # вычислить градиент в v
    v = gradient_step(v, grad, -0.01) # Сделать отрицательный градиентный шаг

print(epoch, v)

assert distance(v, [0, 0, 0]) < 0.001 # v должен быть близким к 0

# Применение градиентного спуска для подгонки моделей
# x измеряется в интервале от-50 до 49, y всегда равно 20 * x + 5
inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

# Линейный градиент
def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta                     # наклон и пересечение
    predicted = slope * x + intercept           # модельное предсказание
    error = (predicted - y)                      # ошибка равна (предсказание - факт)
    squared_error = error ** 2                   # минимизируем квадрат ошибки,
    grad = [2 * error * x, 2 * error]            # используя ее градиент
    return grad

# Задача первый вариант
# начать со случайных значений наклона и пересечения
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

learning_rate = 0.001 # темп усвоения

for epoch in range(5000):
    # вычислить среднее значение градиентов
    grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])
    # сделать шаг в этом направлении
    theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

slope, intercept = theta
assert 19.9 < slope < 20.1, "наклон должен быть равным примерно 20"
assert 4.9 < intercept < 5.1, "пересечение должно быть равным примерно 5"


# Мини-пакетный и стохастический градиентный спуск
T = TypeVar('T') # позволяет типизировать обобщенные функции

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """ Генерирует мини-пакеты в размере 'batch_size' из набора данных"""
    # start индексируется с 0, batch_size, 2 * batch_size,....
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts) # перетасовать пакеты

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]


# Задача второй вариант с мини-пакетами


for epoch in range(1000):
    for batch in minibatches(inputs, batch_size = 20):
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
        theta = gradient_step(theta,grad, -learning_rate)
        print(epoch, v)

slope, intercept = theta
assert 19.9 < slope < 20.1, "наклон должен быть равным примерно 20"
assert 4.9 < intercept < 5.1, "пересечение должно быть равным примерно 5"

# Стохастический ГС таже задача
theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

for epoch in range(100):
    for x, y in inputs:
        grad = linear_gradient(x, y,theta)
        theta = gradient_step(theta, grad, -learning_rate)
        print(epoch,v)

slope, intercept = theta
assert 19.9 < slope < 20.1, "наклон должен быть равным примерно 20"
assert 4.9 < intercept < 5.1, "пересечение должно быть равным примерно 5"

