# ____Тензор____

from typing import List, Callable, Iterable
import operator, random, tqdm, math, mnist, json
from neural_networks import sigmoid
from Probability import inverse_normal_cdf
from LinearAlgebra import dot
from neural_networks import binary_encode, fizz_buzz_encode, argmax
import matplotlib.pyplot as plt

Tensor = list


def shape(tensor: Tensor) -> List[int]:
    '''ищет форму тензора'''
    sizes: List[int] = []
    while isinstance(tensor, list):
        sizes.append(len(tensor))
        tensor = tensor[0]
    return sizes


def is_1d(tensor: Tensor) -> bool:
    """Если tensor[0] является списком, то это тензор более высокого порядка.
    В противном случае tensor является одномерным (вектором)"""
    return not isinstance(tensor[0], list)


def tensor_sum(tensor: Tensor) -> float:
    """суммирует все значения в тензоре"""
    if is_1d(tensor):
        return sum(tensor)
    else:
        return sum(tensor_sum(tensor_i)
                   for tensor_i in tensor)


def tensor_apply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
    """применяет f поэлементно"""
    if is_1d(tensor):
        return [f(x) for x in tensor]
    else:
        return [tensor_apply(f, tensor_i) for tensor_i in tensor]


assert tensor_apply(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
assert tensor_apply(lambda x: 2 * x, [[1, 2], [3, 4]]) == [[2, 4], [6, 8]]


def zeros_like(tensor: Tensor) -> Tensor:
    return tensor_apply(lambda _: 0.0, tensor)


def tensor_combine(f: Callable[[float, float], float],
                   t1: Tensor,
                   t2: Tensor) -> Tensor:
    """ Применяет f к соответствующим элементам тензоров t1 и t2"""
    if is_1d(t1):
        return [f(x, y) for x, y in zip(t1, t2)]
    else:
        return [tensor_combine(f, t1_i, t2_i)
                for t1_i, t2_i in zip(t1, t2)]


assert tensor_combine(operator.add, [1, 2, 3], [4, 5, 6]) == [5, 7, 9]
assert tensor_combine(operator.mul, [1, 2, 3], [4, 5, 6]) == [4, 10, 18]


class Layer:
    """Нейронные сети состоят из слоев, каждый из которых знает,
    как выполнять некоторые вычисления на своих входах в "прямом" направлении
    и распространять градиенты в "обратном" направлении"""

    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradient):
        raise NotImplementedError

    def params(self) -> Iterable[Tensor]:
        """Возвращает параметры этого слоя. Дефолтная имплементация
        ничего не возвращает, потому если слой без параметров, нужно его имплементировать"""
        return ()

    def grads(self) -> Iterable[Tensor]:
        """Возвращает градиенты в том же порядке, что и params()"""
        return ()


class Sigmoid(Layer):
    def forward(self, input: Tensor) -> Tensor:
        """Применить сигмоиду к каждому элементу входного тензора
        и сохранить результаты для использования в обратном распространении"""
        self.sigmoids = tensor_apply(sigmoid, input)
        return self.sigmoids

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda sig, grad: sig * (1 - sig) * grad, self.sigmoids, gradient)


def random_uniform(*dims: int) -> Tensor:
    if len(dims) == 1:
        return [random.random() for _ in range(dims[0])]
    else:
        return [random_uniform(*dims[1:]) for _ in range(dims[0])]


def random_normal(*dims: int,
                  mean: float = 0.0,
                  variance: float = 1.0) -> Tensor:
    if len(dims) == 1:
        return [mean + variance * inverse_normal_cdf(random.random())
                for _ in range(dims[0])]
    else:
        return [random_normal(*dims[1:], mean=mean, variance=variance)
                for _ in range(dims[0])]


assert shape(random_uniform(2, 3, 4)) == [2, 3, 4]
assert shape(random_normal(5, 6, mean=10)) == [5, 6]


def random_tensor(*dims: int,
                  init: str = 'normal') -> Tensor:
    if init == 'normal':
        return random_normal(*dims)
    elif init == 'uniform':
        return random_uniform(*dims)
    elif init == 'xavier':
        variance = len(dims) / sum(dims)
        return random_normal(*dims, variance=variance)
    else:
        raise ValueError(f"unknown init: {init}")


class Linear(Layer):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 init: str = 'xavier') -> None:
        """Слой из dim с input_dim весами каждый
        (and a bias)"""
        self.input_dim = input_dim
        self.output_dim = output_dim

        # self.w[o] - веса для o-го нейрона
        self.w = random_tensor(output_dim, input_dim, init=init)

        # self.b[o] - член смещения o-го нейрона
        self.b = random_tensor(output_dim, init=init)

    def forward(self, input: Tensor) -> Tensor:
        # сохранить вход для использования в обратном прохождении
        self.input = input

        # вернуть вектор выходов нейронов
        return [dot(input, self.w[o]) + self.b[o]
                for o in range(self.output_dim)]

    def backward(self, gradient: Tensor) -> Tensor:
        # каждый b[o] добавляется в output[o], т.е.
        # градиент b тот же самый, что и градиент выхода
        self.b_grad = gradient

        # каждый w[o][i] умножает input[i] и добавляется в output[o].
        # поэтому его градиент равен input[i] * gradient[o]
        self.w_grad = [[self.input[i] * gradient[o]
                        for i in range(self.input_dim)]
                       for o in range(self.output_dim)]

        # каждый input[i] умножает каждый w[o][i] и добавляется в каждый
        # output[o]. поэтому его градиент равен сумме w[o][i] * gradient[o]
        # по всем выходам
        return [sum(self.w[o][i] * gradient[o] for o in range(self.output_dim))
                for i in range(self.input_dim)]

    def params(self) -> Iterable[Tensor]:
        return [self.w, self.b]

    def grads(self) -> Iterable[Tensor]:
        return [self.w_grad, self.b_grad]


class Sequential(Layer):
    """Слой, состоящий из последовательности других слоев.
    Обязательно следить за тем, чтобы выход каждого слоя имел
    смысл в качестве входа в следующий слой."""

    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def forward(self, input):
        """Распространить вход через слои пл порядку"""
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, gradient):
        """Распространить градиент назад через слои в универсуме"""
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def params(self) -> Iterable[Tensor]:
        """вернуть params из каждого слоя"""
        return (param for layer in self.layers for param in layer.params())

    def grads(self) -> Iterable[Tensor]:
        """вернуть grads из каждого слоя"""
        return (grad for layer in self.layers for grad in layer.grads())


class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        """ насколько хорошим является предсказание?"""
        raise NotImplementedError

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """как изменяется потеря при изменении предсказаний?"""
        raise NotImplementedError


class SSE(Loss):
    """функция потери, которая вычисляет сумму квадратов ошибок"""

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        # вычислить тензор квадратических разностей
        squared_errors = tensor_combine(
            lambda predicted, actual: (predicted - actual) ** 2,
            predicted,
            actual)

        # и просто из сложить
        return tensor_sum(squared_errors)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return tensor_combine(
            lambda predicted, actual: 2 * (predicted - actual),
            predicted,
            actual)


class Optimizer:
    """ Оптимизатор обновляет веса слоя (прямо на месте), используя
    информацию, известную либо слою, либо оптимизатору (либо обоим)"""

    def step(self, layer: Layer) -> None:
        raise NotImplementedError


class GradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.1) -> None:
        self.lr = learning_rate

    def step(self, layer: Layer) -> None:
        for param, grad in zip(layer.params(), layer.grads()):
            # обновить param, используя градиентный шаг
            param[:] = tensor_combine(
                lambda param, grad: param - grad * self.lr,
                param,
                grad)


class Momentum(Optimizer):
    def __init__(self,
                 learning_rate: float,
                 momentum: float = 0.9) -> None:
        self.lr = learning_rate
        self.mo = momentum
        self.updates: List[Tensor] = []  # скользящее среднее

    def step(self, layer: Layer) -> None:
        # если нет предыдущих обновлений, то начать со всех нулей
        if not self.updates:
            self.updates = [zeros_like(grad) for grad in layer.grads()]

        for update, param, grad in zip(self.updates,
                                       layer.params(),
                                       layer.grads()):
            # применить импульс
            update[:] = tensor_combine(
                lambda u, g: self.mo * u + (1 - self.mo) * g,
                update,
                grad)

            # затем сделать градиентный шаг
            param[:] = tensor_combine(
                lambda p, u: p - self.lr * u,
                param,
                update)



# _____приемер XOR______
# тренировочные данные
xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
ys = [[0.], [1.], [1.], [0.]]

random.seed(0)

net = Sequential([
    Linear(input_dim=2, output_dim=2),
    Sigmoid(),
    Linear(input_dim=2, output_dim=1)
])

optimizer = GradientDescent(learning_rate=0.1)
loss = SSE()

with tqdm.trange(3000) as t:
    for epach in t:
        epoch_loss = 0.0
        for x, y in zip(xs, ys):
            predicted = net.forward(x)
            epoch_loss += loss.loss(predicted, y)
            gradient = loss.gradient(predicted, y)
            net.backward(gradient)

            optimizer.step(net)

        t.set_description(f"xor потеря {epoch_loss:.3f} ")

for param in net.params():
    print(param)



# ____другие функции активации_____

def tanh(x: float) -> float:
    # если х является очень большим или очень малым,
    # то tanh (по существу) равен -1 или 1.
    if x < -100:
        return - 1
    elif x > 100:
        return 1

    em2x = math.exp(-2 * x)
    return (1 - em2x) / (1 + em2x)


class Tanh(Layer):
    def forward(self, input: Tensor) -> Tensor:
        #  сохранить выход tanh для использования в обратном прохождении
        self.tanh = tensor_apply(tanh, input)
        return self.tanh

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(
            lambda tanh, grad: (1 - tanh ** 2) * grad,
            self.tanh,
            gradient)


class Relu(Layer):
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        return tensor_apply(lambda x: max(x, 0), input)

    def backward(self, gradient: Tensor) -> Tensor:
        return tensor_combine(lambda x, grad: grad if x > 0 else 0,
                              self.input,
                              gradient)


# _______Задача Fizz Buzz_____
xs = [binary_encode(n) for n in range(101, 1024)]
ys = [fizz_buzz_encode(n) for n in range(101, 1024)]

NUM_HIDDEN = 25

random.seed(0)

net = Sequential([
    Linear(input_dim=10, output_dim=NUM_HIDDEN, init='uniform'),
    Tanh(),
    Linear(input_dim=NUM_HIDDEN, output_dim=4, init='uniform'),
    Sigmoid()
])


def fizzbuzz_accuracy(low: int, hi: int, net: Layer) -> float:
    num_correct = 0
    for n in range(low, hi):
        x = binary_encode(n)
        predicted = argmax(net.forward(x))
        actual = argmax(fizz_buzz_encode(n))
        if predicted == actual:
            num_correct += 1

    return num_correct / (hi - low)


optimizer = Momentum(learning_rate=0.1, momentum=0.9)
loss = SSE()

with tqdm.trange(1000) as t:
    for epoch in t:
        epoch_loss = 0.0

        for x, y in zip(xs, ys):
            predicted = net.forward(x)
            epoch_loss += loss.loss(predicted, y)
            gradient = loss.gradient(predicted, y)
            net.backward(gradient)

            optimizer.step(net)

        accuracy = fizzbuzz_accuracy(101, 1024, net)
        t.set_description(f"fb потеря: {epoch_loss:.2f} точность: {accuracy:.2f}")

# проверка результатов на тестовом наборе данных
print("тестовые результаты", fizzbuzz_accuracy(1, 101, net))



# _____softmax и перекретная энтропия(отрацительное логарифмическое правдоподобие)_____

def softmax(tensor: Tensor) -> Tensor:
    """взять softmax вдоль последней размерности"""
    if is_1d(tensor):
        # вычесть наибольшее значение в целях числовой стабильности
        largest = max(tensor)
        exps = [math.exp(x - largest) for x in tensor]
        sum_of_exps = sum(exps)  # суммарный вес
        return [exp_i / sum_of_exps  # вероятность - это доля
                for exp_i in exps]  # суммарного веса
    else:
        return [softmax(tensor_i) for tensor_i in tensor]


class SoftmaxCrossEntropy(Loss):
    """
    Отрицательное логарифмические правлоподобие наблюдаемых значений
    при наличии нейросетевой модели. если выбрать веса для минимизации,
    то модель будет максимизировать правдоподобие наблюдаемых данных.
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        # применить softmax, чтобы получить вероятности
        probabilities = softmax(predicted)

        # будет логарифмом p_i для фактического класса i
        # и 0 для других классов. добавл. маленькое значение
        # в p во избежание взятия log(0)
        likelihoods = tensor_combine(lambda p,
                                            act: math.log(p + 1e-30) * act,
                                     probabilities,
                                     actual)

        # просуммировать отрицательные значения
        return -tensor_sum(likelihoods)

    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        probabilities = softmax(predicted)

        return tensor_combine(lambda p, actual: p - actual,
                              probabilities,
                              actual)



random.seed(0)

net = Sequential([
    Linear(input_dim=10, output_dim=NUM_HIDDEN, init='uniform'),
    Tanh(),
    Linear(input_dim=NUM_HIDDEN, output_dim=4, init='uniform')
])

optimizer = Momentum(learning_rate=0.1, momentum=0.9)
loss = SoftmaxCrossEntropy()

with tqdm.trange(100) as t:
    for epoch in t:
        epoch_loss = 0.0

        for x, y in zip(xs, ys):
            predicted = net.forward(x)
            epoch_loss = loss.loss(predicted, y)
            gradient = loss.gradient(predicted, y)
            net.backward(gradient)

            optimizer.step(net)

        accuracy = fizzbuzz_accuracy(101, 1024, net)
        t.set_description(f"fb потеря: {epoch_loss:.3f} точность: {accuracy:.2f}")

# проверка результатов на тестовом наборе
print("тестовые результаты", fizzbuzz_accuracy(1, 101, net))



class Dropout(Layer):
    def __init__(self, p: float) -> None:
        self.p = p
        self.train = True

    def forward(self, input: Tensor) -> Tensor:
        if self.train:
            # создать маску для нулей и единиц с формой, что и вход,
            # используя указанную вероятность
            self.mask = tensor_apply(
                lambda _: 0 if random.random() < self.p else 1,
                input)
            # умножить на маску для отсева входов
            return tensor_combine(operator.mul, input, self.mask)
        else:
            # во время оценивая просто прошкалировать выходы равномерно
            return tensor_apply(lambda x: x * (1 - self.p), input)

    def backward(self, gradient: Tensor) -> Tensor:
        if self.train:
            # распространять градиенты только там, где mask == 1
            return tensor_combine(operator.mul, gradient, self.mask)
        else:
            raise RuntimeError("не вызывайте backward в тренировочном режиме")


# _______пример набор данных MNIST______

# скачивание данных 60 000 изображений формата 28х28
mnist.temporary_dir = lambda: 'D:\DataScience\StudyDS\mnist'

# обе ф-ии скачивают данные, затем возвращают массив NumPy
# вызов .tolist(), тк "тензоры" - просто списки
train_images = mnist.train_images().tolist()
train_labels = mnist.train_labels().tolist()

fig, ax = plt.subplots(10, 10)

for i in range(10):
    for j in range(10):
        # изобразить каждый снимок в черно-белом цвете и спрятать оси
        ax[i][j].imshow(train_images[10 * i * j], cmap='Greys')
        ax[i][j].xaxis.set_visible(False)
        ax[i][j].yaxis.set_visible(False)

plt.show()

test_images = mnist.test_images().tolist()
test_labels = mnist.test_labels().tolist()

# вычислить среднее пиксельное значение
avg = tensor_sum(train_images) / 60000 / 28 / 28

# перецентрировать(средний пиксел варьируется очень близко к 0), перешкалировать и сгладить
train_images = [[(pixel - avg) / 256 for row in image for pixel in row]
                for image in train_images]
test_images = [[(pixel - avg) / 256 for row in image for pixel in row]
               for image in test_images]


# кодирование с одним активным состоянием
def one_hot_encode(i: int, num_labels: int = 10) -> List[float]:
    return [1.0 if j == i else 0.0 for j in range(num_labels)]


assert one_hot_encode(3) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
assert one_hot_encode(2, num_labels=5) == [0, 0, 1, 0, 0]

train_labels = [one_hot_encode(label) for label in train_labels]
test_labels = [one_hot_encode(label) for label in test_labels]


def loop(model: Layer,
         images: List[Tensor],
         labels: List[Tensor],
         loss: Loss,
         optimizer: Optimizer = None) -> None:
    correct = 0  # отслеживать число правильных предсказаний
    total_loss = 0  # отслеживать суммарную потерю

    with tqdm.trange(len(images)) as t:
        for i in t:
            predicted = model.forward(images[i])  # предсказать
            if argmax(predicted) == argmax(labels[i]):  # проверить на
                correct += 1  # правильность
            total_loss += loss.loss(predicted, labels[i])  # вычислить потерю

            # обучение с распространением градиента по сети назад
            # с обновлением весов
            if optimizer is not None:
                gradient = loss.gradient(predicted, labels[i])
                model.backward(gradient)
                optimizer.step(model)

            # обновить метрики в индикаторе выполнения
            avg_loss = total_loss / (i + 1)
            acc = correct / (i + 1)
            t.set_description(f"mnist потеря: {avg_loss:.3f}  точность: {acc:.3f}")


random.seed(0)

# логистическая регрессия - просто линейный слой,
# за которым следует softmax
model = Linear(784, 10)
loss = SoftmaxCrossEntropy()

optimizer = Momentum(learning_rate=0.01, momentum=0.99)

# обучение
loop(model, train_images, train_labels, loss, optimizer)

# тестирование на тестовых данных
# (отсутствие оптимизатора означает просто оценивание)
loop(model, test_images, test_labels, loss)

# глубокая нейронная сеть
random.seed(0)

# отключать/включать обучение
dropout1 = Dropout(0.1)
dropout2 = Dropout(0.1)

model = Sequential([
    Linear(784, 30),  # скрытый слой 1: размер 30
    dropout1,
    Tanh(),
    Linear(30, 10),  # скрытый слой 2: размер 10
    dropout2,
    Tanh(),
    Linear(10, 10)  # выходной слой: размер 10
])

optimizer = Momentum(learning_rate=0.01, momentum=0.99)
loss = SoftmaxCrossEntropy()

# включить отсев и обучать
dropout1.train = dropout2.train = True
loop(model, train_images, train_labels, loss, optimizer)

# выключить отсев и оценить
dropout1.train = dropout2.train = False
loop(model, test_images, test_labels, loss)

# сохранение и загрузка моделей
def save_weights(model: Layer, filename: str) -> None:
    weights = list(model.params())
    with open (filename, 'w') as f:
        json.dump(weights, f)


def load_weights(model: Layer, filename: str) -> None:
    with open (filename) as f:
        weights = json.load(f)

        # проверить на непротиворечивость
        assert all(shape(param) == shape(weights)
                   for param, weight in zip(model.params(), weights))

    # загрузить, применив срезовое присвоение
    for param, weight in zip(model.params(), weights):
        param[:] = weight