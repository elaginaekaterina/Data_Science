# ___________ПРОСТАЯ ЛИНЕЙНАЯ РЕГРЕССИЯ_________

from LinearAlgebra import Vector
from typing import Tuple
from Statistika import correlation, standard_deviation, mean, \
    num_friends_good, daily_minutes_good, de_mean
import random
import tqdm
from gradient_descent import gradient_step
from matplotlib import pyplot as plt


def predict(alpha: float, beta: float, x_i: float) -> float:
    return beta * x_i + alpha


def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """ошибка предсказания beta * x_i + alpha,
    когда фактическое значение равно y_i"""
    return predict(alpha, beta, x_i) - y_i


def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))


def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
    """учитывая векторы x и y, найти
    значение alpha и beta по наименьшим квадратам"""
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


# тест модели
x = [i for i in range(-100, 100, 10)]
y = [3 * i - 5 for i in x]

# должна отыскать, что y = 3x-5
assert least_squares_fit(x, y) == (-5, 3)

# применение модели к данным без выбросов (очищенным от них предварительно)
alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)

assert 22.9 < alpha < 23.0
assert 0.9 < beta < 0.905

print("alpha=", alpha, '\n', "beta=", beta)


plt.plot(alpha, beta)
plt.scatter(num_friends_good, daily_minutes_good)
plt.title("Простая линейная модель")
plt.xlabel("Число друзей")
plt.ylabel("Число минут на сайте")
plt.show()

def total_sum_of_squares(y: Vector) -> float:
    """полная сумма квадратов отклонений y_i от их среднего"""
    return sum(v ** 2 for v in de_mean(y))


def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    """доля отклонения в y, улавливаемая моделью, которя равна
    '1 - доля отклонения в y, не улавливаемая моделью"""
    return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) /
                  total_sum_of_squares(y))


rsq = r_squared(alpha, beta, num_friends_good, daily_minutes_good)
assert 0.328 < rsq < 0.330
print("raq =", rsq)

# ___________ПРИМЕНЕНИЕ ГРАДИЕНТНОГО СПУСКА__________

num_epochs = 10000
random.seed(0)

guess = [random.random(), random.random()]  # выбрать случайное число для запуска
learning_rate = 0.00001  # темп усвоения

with tqdm.trange(num_epochs) as t:
    for _ in t:
        alpha, beta = guess

        # частная производная потери по отношению к alpha
        grad_a = sum(2 * error(alpha, beta, x_i, y_i)
                     for x_i, y_i in zip(num_friends_good, daily_minutes_good))

        # частная производная потери по отношению к beta
        grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i
                     for x_i, y_i in zip(num_friends_good, daily_minutes_good))

    # вычислить потерю для вставки в описание tqdm
    loss = sum_of_sqerrors(alpha, beta, num_friends_good, daily_minutes_good)
    t.set_description(f"потеря:{loss:.3f}")

    # обновить догадку
    guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)

# результаты должны быть практически одинаковые стр.48:
alpha, beta = guess
assert 22.9 < alpha < 23.0
assert 0.9 < beta < 0.905
