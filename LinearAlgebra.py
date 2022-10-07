# ________ Векторы________
from typing import List
import math
from typing import Tuple
from typing import Callable

Vector = List[float]


# сложение векторов + функция zip
def add(v: Vector, w: Vector) -> Vector:
    """Складывем соответствующие элементы"""
    assert len(v) == len(w)  # проверка на одинаковую длину

    return [v_i + w_i for v_i, w_i in zip(v, w)]


assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


# вычитание векторов
def subtract(v: Vector, w: Vector) -> Vector:
    """Вычитаем соответствующие элементы"""
    assert len(v) == len(w)  # проверка на одинаковую длину

    return [v_i - w_i for v_i, w_i in zip(v, w)]


assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]


# покомпонентная сумма списка векторов
def vector_sum(vectors: List[Vector]) -> Vector:
    """Суммирует все соответствующие элементы"""
    assert vectors, "Векторы не предоставлены!"

    # проверить, что все векторы имеют одинаковый размер
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "разные размеры!"

    # i-ый элемент результата является суммой каждого элемента vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]


assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]


# умножение вектора на скаляр
def scalar_multiply(c: float, v: Vector) -> Vector:
    """Умножает каждый элемент на c"""
    return [c * v_i for v_i in v]


assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]


# вычисления покомпонентных средних значений списка векторов
# одинакового размера
def vector_mean(vectors: List[Vector]) -> Vector:
    """Вычисляет поэлементное среднее значение"""
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))


assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]


# Скалярное произведение двух векторов(сумма покомпонентных произведений)
def dot(v: Vector, w: Vector) -> float:
    """вычисляет v_i * w_i + ...+v_n * w_n
    :rtype: object
    """
    assert len(v) == len(w)

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


assert dot([1, 2, 3], [4, 5, 6]) == 32


# сумма квадратов вектора для вычисления магнитуды (длины) ветора
def sum_of_squares(v: Vector) -> float:
    """возвращает v_1 * v_1 +...+v_n * V_n"""
    return dot(v, v)


assert sum_of_squares([1, 2, 3]) == 14


def magnitude(v: Vector) -> float:
    """вызврящает магнитуду вектрра v"""
    return math.sqrt(sum_of_squares(v))


def squared_distance(v: Vector, w: Vector) -> float:
    """вычисляет (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))


# Евклидово расстояние
def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))


print(distance([1, 2, 3], [4, 5, 6]))

# _________Матрицы___________
Matrix = List[List[float]]


def shape(A: Matrix) -> Tuple[int, int]:
    """возвращает число строк А, число столбцов А"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0  # число элементов в первой строке
    return num_rows, num_cols


assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)  # 2 строки, 3 столбца


def get_row(A: Matrix, i: int) -> Vector:
    """Возвращает i-ую строку А (как тип Vector)"""
    return A[i]


def get_col(A: Matrix, j: int) -> Vector:
    """Возвращает j-ый столбец А (как тип Vector)"""
    return [A_i[j]  # j-ый элемент A_i
            for A_i in A]  # для каждой строки A_i


# вложенное включение в список для создания матрицы
# на снове ее формы и функции, что задает ее эл-ты
def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Возвращает матрицу размера num_rows х num_cols,
    чей (i,j)-ый элемент является функцией entry_fn(i,j)
    """
    return [[entry_fn(i, j)  # Создание списка с учетом i
             for j in range(num_cols)]  # [entry_fn(i, 0),...]
            for i in range(num_rows)]  # создание одного списка для каждого i


def identity_matrix(n: int) -> Matrix:
    """возвращает (n x n) - матрицу тождественности (единичную)"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]
