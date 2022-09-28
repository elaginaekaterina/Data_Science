#________________ГИПОТЕЗЫ____________________

from typing import Tuple
import math
from Probability import normal_cdf

# Аппроксимация биномиальной СВ нормальным распределением
def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """Возвращает mu и sigma, соответствующие binomial(n, p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 -p) * n)
    return mu, sigma

# Нормальная функция CDF - вероятность, что
#  переменная лежит ниже порога
normal_probability_below = normal_cdf

# она ледит ниже порога, если она не ниже порога
def normal_probability_above(lo: float,
                             mu: float = 0,
                             sigma: float = 1) -> float:
    """ вероятность, что N(mu, sigma) выше чем lo."""
    return 1 -normal_cdf(lo, mu, sigma)

# она лежит между, если она меньше, чем hi, но не меньше чем lo
def normal_probability_between(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    """вероятность,что N(mu, sigma) между lo и hi"""
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# она лежит за пределами, если она не лежит между
def normal_probability_outside(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    """вероятность, что N(mu, sigma) нележит между lo и hi"""
    return 1 - normal_probability_between(lo, mu, sigma)

from Probability import inverse_normal_cdf

# верхняя граница
def normal_upper_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """возвращает z, для которой P(Z <= z) = вероятность"""
    return inverse_normal_cdf(probability, mu, sigma)

# нижняя граница
def normal_lower_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """возвращает z, для которой P(Z >= z) = вероятность"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

# двусторонняя граница
def normal_two_sided_bounds(probability: float,
                           mu: float = 0,
                           sigma: float = 1) -> float:
    """возвращает симметрические границы,
    котрорые содержат указанную вероятность"""
    tail_probability = (1 - probability) / 2

    # верхняя граница должна иметь хвостовую tail_probability выше ее
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # нижняя граница должна иметь хвостовую tail_probability ниже ее
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    return lower_bound,upper_bound

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

# (469, 531), ошибка 1-го рода = 5%
lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# 95%-ые границы, основанные на допущении, что p = 0.5
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# фактические mu и sigma, основанные на p = 0.55
mu_1, sigma_1 =normal_approximation_to_binomial(1000, 0.55)

# ошибка 2-го рода означает, что нам не удалось отклонить нулевуб гипотезу,
# что произойдет, когда X все еще внутри нашего исходного интервала
type_2_probability_1 = normal_probability_between(lo, hi, mu_1, sigma_1)
power_1 = 1 - type_2_probability_1

hi = normal_upper_bound(0.95, mu_0, sigma_0)
# равно 526(< 531, т.к. нужно больше вероятности в верхнем хвосте

type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power_2 = 1 -type_2_probability

print(power_1, power_2)

#__________P-значения____________
# двустороннее p-значение
def two_sided_p_value(x: float, mu: float = 0,
                      sigma: float = 1) -> float:
    """ насколько правдоподобно увидеть значение, как минимум, такое же
    предельное, что и x (в любом направлении), если наши значения
    поступают из N(mu, sigma)?"""
    if x >= mu:
        # x больше, чем среднее, поэтому хвост везде больше, чем x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # x меньше, чем среднее, поэтом хвост везде меньше, чем x
        return 2 * normal_probability_below(x, mu, sigma)

two_sided_p_value(529.5, mu_0, sigma_0)

# симуляция
import random

extreme_value_count = 0
for _ in range(1000):
    num_heads = sum(1 if random.random() < 0.5 else 0     # подсчитать число орлов
                    for _ in range(1000))                 # в 1000 бросках
    if num_heads >= 530 or num_heads <= 470:              # и как часто это число
        extreme_value_count += 1                          # 'предельное'

# p - значение было 0.062 ~ 62 предельных значений из 1000
assert 59 < extreme_value_count < 65, f"{extreme_value_count}"

# получить вернее и нижнее p-значение:
upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

# односторонняя проверка при x = 525
upper_p_value(524.5, mu_0, sigma_0)  # 0.061 и H0 - не отклонена

# x = 527
upper_p_value(526.5, mu_0, sigma_0)  # 0.047 и H0 отклонена


#__________________Взлом p-значения_____________
from typing import List

def run_experiment() -> List[bool]:
    """ подбрасывает уравновешенную монету 1000 раз,
    Истина = орлы, Ложь = решки"""
    return [random.random() < 0.5 for _ in range(1000)]

def reject_fairness(experiment: List[bool]) -> bool:
    """использование 5%-ых уровней хзначимости"""
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531

random.seed(0)
experiments = [run_experiment() for _ in range(1000)]

num_rejections = len([experiment
                      for experiment in experiments
                      if reject_fairness(experiment)])

assert num_rejections == 46


# Проведение A/B тестирования
# оценочные параметры
def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma

def a_b_test_statistic(N_A: int, n_A:int,
                       N_B: int, n_B: int) -> float:
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

# пример A = 200 из 1000, B = 180  из 1000
z = a_b_test_statistic(1000, 200, 1000, 180)
print(z)

print(two_sided_p_value(z))

# Байесов вывод
def B(alpha: float, beta: float) -> float:
    """нормализующая константа, чтобы полная вроятность
    в сумме составляла 1"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x: float, alpha: float, beta: float) -> float:
    if x <= 0 or x >= 1:   #за пределами [0, 1] нет веса
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)