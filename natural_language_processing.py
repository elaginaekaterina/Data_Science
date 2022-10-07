import json
import math
import random
import re
import tqdm
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup
import requests
from collections import Counter
from collections import defaultdict
from typing import List, Dict, Tuple, Iterable

from LinearAlgebra import dot, Vector
import deep_learning
from working_with_data import pca, transform
from deep_learning import softmax, SoftmaxCrossEntropy, Momentum

data = [("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
        ("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
        ("data science", 60, 70), ("analytics", 90, 3),
        ("team player", 85, 85), ("dynamic", 2, 90), ("synergies", 70, 0),
        ("actionable insights", 40, 30), ("think out of the box", 45, 10),
        ("self-starter", 30, 50), ("customer focus", 65, 15),
        ("thought leadership", 35, 35)]


def text_size(total: int) -> float:
    """Равняется 8, есди итог равен 0, и 28, если итог равен 200"""
    return 8 + total / 200 * 20


for word, job_popularity, resume_popularity in data:
    plt.text(job_popularity, resume_popularity, word,
             ha='center', va='center',
             size=text_size(job_popularity + resume_popularity))
plt.xlabel("Популярность среди объявлений о вакансиях")
plt.ylabel("Популярность  среди резюме")
plt.axis([0, 100, 0, 100])
plt.show()


# _______N-граммные языковые модели______
def fix_unicode(text: str) -> str:
    return text.replace(u"\u2019", "'")


url = "https://www.oreilly.com/ideas/what-is-data-science"
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

content = soup.find("div", "article-body")  # отыскать div с именем article-body
regex = r"[\w] + |[\.]"  # сочетает слово и точку

document = []

for paragraph in content("p"):
    words = re.findall(regex, fix_unicode(paragraph.text))
    document.extend(words)

transitions = defaultdict(list)
for prev, current in zip(document, document[1:]):
    transitions[prev].extend(current)


def generate_using_bigrams() -> str:
    current = "."  # означает, что следующее слово начнет предложение
    result = []
    while True:
        next_word_candidates = transitions[current]  # биграммы (current, _)
        current = random.choice(next_word_candidates)  # выбрать одно случайно
        result.extend(current)  # добавить result
        if current == ".": return " ".join(result)  # если ".", то завершить


trigram_transition = defaultdict(list)
starts = []

for prev, current, next in zip(document, document[1:], document[2:]):
    if prev == ".":  # если предыдущее "слово" было точкой,
        starts.append(current)  # то это стартовое слово

    trigram_transition[(prev, current)].append(next)


def generate_using_trigrams() -> str:
    current = random.choice(starts)  # выбрать случайное стартовое слово
    prev = "."  # и предварить его символом "."
    result = [current]
    while True:
        next_word_candidates = trigram_transition[(prev, current)]
        next_word = random.choice(next_word_candidates)

        prev, current = current, next_word
        result.append(current)

        if current == ".":
            return " ".join(result)


# ________Грамматики________

# псевдоним слов для ссылки на грамматики позже
Grammar = Dict[str, List[str]]

grammar = {
    "_S": ["_NP _VP"],
    "_NP": ["_N",
            "_A _NP _P _A _N"],
    "_VP": ["_V",
            "_V _NP"],
    "_N": ["data science", "Python", "regressions"],
    "_A": ["big", "linear", "logostic"],
    "_P": ["about", "near"],
    "_V": ["learns", "trains", "tests", "is"]
}


def is_terminal(token: str) -> bool:
    return token[0] != "_"


def expand(grammar: Grammar, tokens: List[str]) -> List[str]:
    for i, token in enumerate(tokens):
        # если это терминальная лексема, то пропустить ее
        if is_terminal(token): continue
        # в противном случае это нетерминальная лексема,
        # поэтому нужно случайно выбрать подстановку
        replacement = random.choice(grammar[token])

        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            # подстановкой могло быть, например, "_NP _VP", поэтому
            # нужно разбить по пробелам и присоединить к списку
            tokens = tokens[:i] + replacement.split() + tokens[(i + 1):]

        # вызвать expand с новым списком лексем
        return expand(grammar, tokens)

    # закончить обработку, если все терминалы
    return tokens


def generate_sentence(grammar: Grammar) -> List[str]:
    return expand(grammar, ["_S"])



# ______Генерирование выборок по Гиббсу_______
def roll_a_die() -> int:
    return random.choice([1, 2, 3, 4, 5, 6])


def direct_sample() -> Tuple[int, int]:
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1 + d2


def random_y_given_x(x: int) -> int:
    """равновероятно составляет x+1, x+2,...,x+6"""
    return x + roll_a_die()


def random_x_given_y(y: int) -> int:
    if y <= 7:
        # если сумма <= 7, то первый кубик равновероятно будет
        # 1, 2, ..., (сумма - 1)
        return random.randrange(1, y)
    else:
        # если сумма > 7, то первый кубик равновероятно будет
        # (сумма - 6), (сумма - 5), ..., 6
        return random.randrange(y - 6, 7)


# выборка по Гиббсу
def gibbs_sample(num_iters: int = 100) -> Tuple[int, int]:
    x, y = 1, 2
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y


# сравнить распределения
def compare_distributions(num_samples: int = 1000) -> Dict[int, List[int]]:
    counts = defaultdict(lambda: [0, 0])
    for _ in range(num_samples):
        counts[gibbs_sample()][0] += 1
        counts[direct_sample()][1] += 1
    return counts


# ________ТЕМАТИЧЕСКОЕ МОДЕЛИРОВАНИЕ_______
def sample_from(weights: List[float]) -> int:
    """"возвращает i с вероятностью weights[i] / sum(weights)"""
    total = sum(weights)
    rnd = total * random.random()  # равномерно между 0 и суммой
    for i, w in enumerate(weights):
        rnd -= w  # вернуть наименьший i, такой, что
        if rnd <= 0: return i  # weights[0] + ... + weights[i] >= rnd


# извлечь 1000 раз и подсчитать
draws = Counter(sample_from([0.1, 0.1, 0.8]) for _ in range(1000))
assert 10 < draws[0] < 190
assert 10 < draws[1] < 190
assert 10 < draws[2] < 950
assert draws[0] + draws[1] + draws[2] == 1000

documents = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

K = 4

# Список объектов Counter, один для каждого документа
document_topic_counts = [Counter() for _ in documents]

# Список объектов Counter, один для каждой тематики
topic_word_counts = [Counter() for _ in range(K)]

# Список чисел, один для каждой тематики
topic_counts = [0 for _ in range(K)]

# Список чисел, один для каждого документа
document_lengths = [len(document) for document in documents]

distinct_words = set(word for document in documents for word in document)
W = len(distinct_words)
D = len(documents)


# вероятность тематики в зависимости от документа
def p_topic_given_document(topic: int, d: int,
                           alpha: float = 0.1) -> float:
    """доля слов в документе 'd', которые назначаются тематике
    'topic' (плюс некоторое сглаживание)"""

    return ((document_topic_counts[d][topic] + alpha) /
            (document_lengths[d] + K * alpha))


# вероятность слова в зависимости от тематики
def p_word_given_topic(word: str, topic: int,
                       beta: float = 0.1) -> float:
    """доля слов, назначаемых тематике 'topic', которые
    равны 'word' (плюс некоторое сглаживание)"""

    return ((topic_word_counts[topic][word] + beta) /
            (topic_counts[topic] + W * beta))


# вес тематики
def topic_weight(d: int, word: str, k: int) -> float:
    """с учетом документа и слова в этом документе,
    вернуть вес k-й тематики"""
    return p_word_given_topic(word, k) * p_topic_given_document(k, d)


# выбор новой тематики
def choose_new_topic(d: int, word: str) -> int:
    return sample_from([topic_weight(d, word, k)
                        for k in range(K)])


random.seed(0)
document_topics = [[random.randrange(K) for word in document]
                   for document in documents]

for d in range(D):
    for word, topic in zip(documents[d], document_topics[d]):
        document_topic_counts[d][topic] += 1
        topic_word_counts[topic][word] += 1
        topic_counts[topic] += 1

for iter in tqdm.trange(1000):
    for d in range(D):
        for i, (word, topic) in enumerate(zip(documents[d],
                                              document_topics[d])):
            # удалить это слово/тематику из показателя,
            # чтобы оно не влияло на веса
            document_topic_counts[d][topic] -= 1
            topic_word_counts[topic][word] -= 1
            topic_counts[topic] -= 1
            document_lengths[d] -= 1

            # выбрать новую тематику, основываясь на весах
            new_topic = choose_new_topic(d, word)
            document_topics[d][i] = new_topic

            # и добавить его назад в показатель
            document_topic_counts[d][new_topic] += 1
            topic_word_counts[topic][word] += 1
            topic_counts[new_topic] += 1
            document_lengths[d] += 1

# просмотр наиболее распространенных слов ы тематике
for k, word_counts in enumerate(topic_word_counts):
    for word, count in word_counts.most_common():
        if count > 0:
            print(k, word, count)

topic_names = ["Большие данные и языка программирования",
               "Базы данных",
               "Машинное обучение",
               "Python и статистика"]

# каким образом модель назначает тематики
for document, topic_counts in zip(documents, document_topic_counts):
    print(document)
    for topic, count in topic_counts.most_common():
        if count > 0:
            print(topic_names[topic], count)
    print()


# _________ВЕКТОРЫ СЛОВ + НС________

def cosine_similarity(v1: Vector, v2: Vector) -> float:
    return dot(v1, v2) / math.sqrt(dot(v1, v1) * dot(v2, v2))


assert cosine_similarity([1., 1, 1], [2., 2, 2]) == 1
assert cosine_similarity([-1., -1], [2., 2]) == -1
assert cosine_similarity([1., 0], [0., 1]) == 0

# искусственный набор данных
colors = ["red", "green", "blue", "yellow", "bleck", ""]
nouns = ["bed", "car", "boat", "cat"]
verbs = ["is", "was", "seems"]
adverbs = ["very", "quite", "extremely", ""]
adjectives = ["slow", "fast", "soft", "hard"]


# составить предложение
def make_sentence() -> str:
    return " ".join([
        random.choice(colors),
        random.choice(nouns),
        random.choice(verbs),
        random.choice(adverbs),
        random.choice(adjectives),
        "."
    ])


NUM_SENTENCES = 50

random.seed(0)
sentences = [make_sentence() for _ in range(NUM_SENTENCES)]


class Vocabulary:
    def __init__(self, words: List[str] = None) -> None:
        self.w2i: Dict[str, int] = {}  # отображение word -> word_id
        self.i2w: Dict[int, str] = {}  # отображение word_id -> word

        for word in (words or []):  # если слова были предоставлены,
            self.add(word)  # то добавить их

    @property
    def size(self) -> int:
        """сколько слов в лексиконе"""
        return len(self.w2i)

    def add(self, word: str) -> None:
        if word not in self.w2i:  # если слово новое:
            word_id = len(self.w2i)  # то отыскать следующий id
            self.w2i[word] = word_id  # добавить в отображение word -> word_id
            self.i2w[word_id] = word  # добавить в отображение word_id -> word

    def get_id(self, word: str) -> int:
        """вернуть id слова (либо None)"""
        return self.w2i.get(word)

    def get_word(self, word_id: int) -> str:
        """вернуть слово с заданным id (либо None)"""
        return self.i2w.get(word_id)

    def one_hot_encode(self, word: str) -> deep_learning.Tensor:
        word_id = self.get_id(word)
        assert word_id is not None, f"неизвестное слово {word}"

        return [1.0 if i == word_id else 0.0 for i in range(self.size)]


# вспомогательные функции для сохранения и загрузки словаря

def save_vocab(vocab: Vocabulary, filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(vocab.w2i, f)  # нужно сохранить только w2i


def load_vocab(filename: str) -> Vocabulary:
    vocab = Vocabulary()
    with open(filename) as f:
        # загрузить w2i и сгенерировать из него i2w
        vocab.w2i = json.load(f)
        vocab.i2w = {id: word for word, id in vocab.w2i.items()}
    return vocab


# создание НС
# 1. слой вложения (общий)
class Embedding(deep_learning.Layer):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # один вектор размера embedding_dim для каждого желаемого вложения
        self.embeddings = deep_learning.random_tensor(num_embeddings, embedding_dim)
        self.grad = deep_learning.zeros_like(self.embeddings)

        # сохранить id последнего входа
        self.last_input = None

    def forward(self, input_id: int) -> deep_learning.Tensor:
        """просто выбрать вектор вложения, соответствующий входному id"""
        self.input_id = input_id  # запомнить для использования
        # в обратном распространении
        return self.embeddings[input_id]

    def backward(self, gradient: deep_learning.Tensor) -> None:
        # обнулить градиент, соответствующий последнему входу.
        if self.last_input_id is not None:
            zero_row = [0 for _ in range(self.embedding_dim)]
            self.grad[self.last_input_id] = zero_row

        self.last_input_id = self.input_id
        self.grad[self.input_id] = gradient

    def params(self) -> Iterable[deep_learning.Tensor]:
        return [self.embeddings]

    def grads(self) -> Iterable[deep_learning.Tensor]:
        return [self.grad]


# 2. подкласс для векторов слов
class TextEmbedding(Embedding):
    def __init__(self, vocab: Vocabulary, embedding_dim: int) -> None:
        # вызвать конструктор суперкласса
        super().__init__(vocab.size, embedding_dim)

        # зафиксировать лексикон
        self.vocab = vocab

    def __getitem__(self, word: str) -> deep_learning.Tensor:
        word_id = self.vocab.get_id(word)
        if word_id is not None:
            return self.embeddings[word_id]
        else:
            return None

    def closest(self, word: str, n: int = 5) -> List[Tuple[float, str]]:
        """возвращает n ближайших слов на основе косинусного сходства"""
        vector = self[word]

        # вычислить пары (сходство, другое_сходство) и отсортировать по схожести
        # (самое сложное идет первым)
        scores = [(cosine_similarity(vector, self.embeddings[i]), other_word)
                  for other_word, i in self.vocab.w2i.items()]
        scores.sort(reverse=True)

        return scores[:n]


tokenized_sentences = [re.findall("[a-z] + |[.]", sentences.lower())
                       for sentence in sentences]

# создать лексикон (т.е. отображение word -> word_id)
# на основе текста
vocab = Vocabulary(word
                   for sentence_words in tokenized_sentences
                   for word in sentence_words)

# создание тренировочных данных
inputs: List[int] = []
targets: List[deep_learning.Tensor] = []

for sentence in tokenized_sentences:
    for i, word in enumerate(sentence):    # для каждого слова
        for j in [i - 2, i - 1, i + 1, i + 2]:    # взять ближайшие расположения,
            if 0 <= j < len(sentence):    # которые не выходят за границы,
                nearby_word = sentence[j]    # и получить эти слова

                # добавить вход, т.е. исходный word_id
                inputs.append(vocab.get_id(word))

                # добавить цель, т.е. близлежащее слово,
                # кодированное с одним активным состоянием
                targets.append(vocab.one_hot_encode(nearby_word))


random.seed(0)

EMBEDDING_DIM = 5

# определить слой вложения отдельно,
# чтобы можно было на него ссылаться
embedding = TextEmbedding(vocab = vocab, embedding_dim=EMBEDDING_DIM)

model = deep_learning.Sequential([
    # с учетом заданного слова (как вектора идентификаторов word_id)
    # найти его вложение
    embedding,
    # и применить линейный слой для вычисления
    # балльных отметок для "близлежащих слов"
    deep_learning.Linear(input_dim = EMBEDDING_DIM, output_dim = vocab.size)
])

loss = deep_learning.SoftmaxCrossEntropy()
optimizer = deep_learning.GradientDescent(learning_rate = 0.01)

for epoch in range(100):
    epoch_loss = 0.0
    for input, target in zip(inputs, targets):
        predicted = model.forward(input)
        epoch_loss = loss.loss(predicted, target)
        gradient = loss.gradient(predicted, target)
        model.backward(gradient)
        optimizer.step(model)

    print(epoch, epoch_loss)
    print(embedding.closest("black"))
    print(embedding.closest("slow"))
    print(embedding.closest("car"))


# извлечь первые главные компоненты и преобразовать в векторы слов
components = pca(embedding.embeddings, 2)
transformed = transform(embedding.embeddings, components)

# рассеять точки (и сделать их белыми, чтобы они были "невидимыми")
fig, ax = plt.subplots()
ax = plt.scatter(*zip(*transformed), marker = '.', color = 'w')

# добавить аннотации каждого слова в его
# трансформированном расположении
for word, idx in vocab.w2i.items():
    ax.annotate(word, transformed[idx])

# спрятать оси
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()


#_________РЕКУРРЕНТНЫЕ НЕЙРОННЫЕ СЕТИ________
class SimpleRNN(deep_learning.Layer):
    """почти простейший из возможных рекуррентный слой"""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.w = deep_learning.random_tensor(hidden_dim, input_dim, init='xavier')
        self.u = deep_learning.random_tensor(hidden_dim, hidden_dim, init='xavier')
        self.b = deep_learning.random_tensor(hidden_dim)

        self.reset_hidden_state()

    def reset_hidden_state(self) -> None:
        self.hidden = [0 for _ in range(self.hidden_dim)]

    def forward(self, input: deep_learning.Tensor) -> deep_learning.Tensor:
        self.input = input    # сохранить вход и предыдущее скрытое
        self.prev_hidden = self.hidden    # состояние для обратного распространения

        a = [(dot(self.w[h], input) +    # веса @ вход
              dot(self.u[h], self.hidden) +     # веса @ скрытое состояние
              self.b[h])     # смещение
             for h in range(self.hidden_dim)]

        self.hidden = deep_learning.tensor_apply(deep_learning.tanh, a)    # применить активацию tanh
        return self.hidden    # вернуть результат

    def backward(self, gradient: deep_learning.Tensor):
        # распространить назад через активацию tanh
        a_grad = [gradient[h] * (1 - self.hidden[h] ** 2)
                  for h in range(self.hidden_dim)]

        # b имеет тот же градиент, что и a
        self.b = a_grad

        # каждый w[h][i] умножается на input[i] и добавляется в a[h],
        # поэтому каждый w_grad[h][i] = a_grad[h] * input[i]
        self.w_grad = [[a_grad[h] * self.input[i]
                        for i in range(self.input_dim)]
                       for h in range(self.hidden_dim)]

        # каждый u[h][h2] умножается на hidden[h2] и добавляется в a[h],
        # поэтому каждый u_grad[h][h2] = a_grad[h] * prev_hidden[h2]
        self.u_grad = [[a_grad[h] * self.prev_hidden[h2]
                        for h2 in range(self.hidden_dim)]
                       for h in range(self.hidden_dim)]

        # каждый input[i] умножается на каждый w[h][i] и добавляется
        # в a[h], поэтому каждый
        # input_grad[i] = sum(a_grad[h] * w[h][i] for h in ...)
        return [sum(a_grad[h] * self.w[h][i] for h
                    in range(self.hidden_dim))
                for i in range(self.input_dim)]

    def params(self) -> Iterable[deep_learning.Tensor]:
        return [self.w, self.u, self.b]

    def grads(self) -> Iterable[deep_learning.Tensor]:
        return [self.w_grad, self.u_grad, self.b_grad]


# пример: использование RNN-сети уровня букв
url = "https://www.ycombinator.com/topcompanies/"
soup = BeautifulSoup(requests.get(url).text, "html5lib")

# применение операции включения во множество для удаления повторов
companies = list({b.text
                  for b in soup('b')
                  if "h4" in b.get("class", ())})

assert len(companies) == 101

# построение лексикона из букв в названиях
vocab = Vocabulary([c for company in companies for c in company])

START = "^"
STOP = "&"

vocab.add(START)
vocab.add(STOP)

HIDDEN_DIM = 32

rnn1 = SimpleRNN(input_dim = vocab.size, hidden_dim = HIDDEN_DIM)
rnn2 = SimpleRNN(input_dim = HIDDEN_DIM, hidden_dim = HIDDEN_DIM)
linear = deep_learning.Linear(input_dim = HIDDEN_DIM, output_dim = vocab.size)

model = deep_learning.Sequential([
    rnn1,
    rnn2,
    linear
])

def generate(seed: str = START, max_len: int = 50) -> str:
    rnn1.reset_hidden_state()    # обнулить оба скрытых состояния
    rnn2.reset_hidden_state()
    output = [seed]    # инициализировать выход заданным начальным состоянием

    # продолжать до тех пор, пока не произведется буква STOP
    # либо не достигнется максимальная длина
    while output[-1] != STOP and len(output) < max_len:
        # использовать последнюю букву как вход
        input = vocab.one_hot_encode(output[-1])

        # сгенерировать отметки, используя модель
        predicted = model.forward(input)

        # конвертировать их в вероятности и извлечь случайный char_id
        probabilities = softmax(predicted)
        next_char_id = sample_from(probabilities)

        # добавить соответствующий символ char в выход
        output.append(vocab.get_word(next_char_id))

    # отбросить буквы START и STOP и вернуть слово
    return ''.join(output[1:-1])


loss = SoftmaxCrossEntropy()
optimizer = Momentum(learning_rate = 0.01, momentum = 0.9)

for epoch in range(300):
    random.shuffle(companies)    # тренировать в другом порядке в каждой эпохе

    epoch_loss = 0     # отслеживать потерю
    for company in tqdm.tqdm(companies):
        rnn1.reset_hidden_state()
        rnn2.reset_hidden_state()   # обнулить оба скрытых состояния
        company = START + company + STOP    # добавить буква START и STOP

    # выходы и цель - кодированные в форме с одним активным состоянием предыдущая и следующая буквы
    for prev, next in zip(company, company[1:]):
        input = vocab.one_hot_encode(prev)
        target = vocab.one_hot_encode(next)
        predicted = model.forward(input)
        epoch_loss += loss.loss(predicted, target)
        gradient = loss.gradient(predicted, target)
        model.backward(gradient)
        optimizer.step(model)

    # для каждой эпохи печатать потерю и генерировать название
    print(epoch, epoch_loss, generate())

    # уменьшить темп усвоения для последних 100 эпох.
    if epoch == 200:
        optimizer.lr *= 0.1