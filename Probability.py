import enum
import random


# _________ Условная вероятность________
# Enum - типизированное множество перечислимых значений.
# Используется для описательности и читабельности кода

class Kid(enum.Enum):
    BOY = 0
    GIRL = 1


def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])


both_girls = 0
older_girl = 0
either_girl = 0

random.seed(0)

for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    if older == Kid.GIRL:
        older_girl += 1
    if older == Kid.GIRL and younger == Kid.GIRL:
        both_girls += 1
    if older == Kid.GIRL or younger == Kid.GIRL:
        either_girl += 1

print("P(both | older):", both_girls / older_girl)
print("both | either)", both_girls / either_girl)


# __________Непрерывные распредеения____________

# функция плотности вероятности равномерного распределения:
def uniform_pdf(x: float) -> float:
    return 1 if x >= 0 and x < 1 else 0


# кумулятивная функция распределения
def uniform_cdf(x: float) -> float:
    """Возвращает вероятность, что равномерно
    распределенная случайная величина <= x"""
    if x < 0:
        return 0  # равномерная величина никогда не бывает меньше 0
    elif x < 1:
        return x
    else:
        return 1


# ______________Нормальное распределение______________
import math
import matplotlib.pyplot as plt

SQRT_TWO_PI = math.sqrt(2 * math.pi)


def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return math.exp(-(x - mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma)


'''
xs = [x / 10.0 for x in range (-50, 50)]
plt.plot(xs, [normal_pdf(x, sigma = 1) for x in xs], '-', label = "мю = 0, сигма = 1")
plt.plot(xs, [normal_pdf(x, sigma = 2) for x in xs], '--', label = "мю = 0, сигма = 2")
plt.plot(xs, [normal_pdf(x, sigma = 0.5) for x in xs], ':', label = "мю = 0, сигма = 0.5")
plt.plot(xs, [normal_pdf(x, mu = -1) for x in xs], '-.', label = "мю = -1, сигма = 1")
plt.legend()
plt.title("Различные нормальные функции плотности вероятности")
plt.show()
'''


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


'''
xs = [x / 10.0 for x in range (-50, 50)]
plt.plot(xs, [normal_cdf(x, sigma = 1) for x in xs], '-', label = "мю = 0, сигма = 1")
plt.plot(xs, [normal_cdf(x, sigma = 2) for x in xs], '--', label = "мю = 0, сигма = 2")
plt.plot(xs, [normal_cdf(x, sigma = 0.5) for x in xs], ':', label = "мю = 0, сигма = 0.5")
plt.plot(xs, [normal_cdf(x, mu = -1) for x in xs], '-.', label = "мю = -1, сигма = 1")
plt.legend(loc = 4) # внизу справа
plt.title("Различные нормальные кумулятивные функции распределения")
plt.show()
'''


def inverse_normal_cdf(p: float,
                       mu: float = 0,
                       sigma: float = 1,
                       tolerance: float = 0.00001) -> float:  # задать точность
    """Отыскать приближенную инверсию, используя бинарный поиск"""
    # если не стандартная, то вычислить стандартную и перешкалировать
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    low_z = -10.0  # normal_cdf(-10) равно (находится очень близко к) 0
    hi_z = 10.0  # normal_cdf(10) равно (находится очень близко к) 1

    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2  # рассмотреть среднюю точку
        mid_p = normal_cdf(mid_z)  # и значение CDF
        if mid_p < p:
            low_z = mid_z  # средняя точна слишком низкая, искать выше
        else:
            hi_z = mid_z  # средняя точка лишком высокая, искать ниже
    return mid_z


# _______________Центральная предельная теорема____________
# биномиальная СВ и распределение Бернулли

def bernoulli_trial(p: float) -> int:
    """Возвращает 1 с вероятностью p и 0 с вероятностью 1-p"""
    return 1 if random.random() < p else 0


def binomial(n: int, p: float) -> int:
    """Возвращает сумму из n испытаний bernoulli(p)"""
    return sum(bernoulli_trial(p) for _ in range(n))


from collections import Counter


def binomial_histogram(p: float, n: int, num_points: int) -> None:
    """Подбирает точки из binomial(n, p) и строит гистограмму"""
    data = [binomial(n, p) for _ in range(num_points)]

    # использовать столбчатый график
    # для показа фактических биномиальных выборок
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')

    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))

    # использовать линейный график для показа нормальной аппроксимации
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)
          for i in xs]

    plt.plot(xs, ys)
    plt.title("Биномиальное распределение и его нормальное приближение")
    plt.show()
