# ________1. Наивный Байесов классификатор (спам-фильтр)________

from typing import Set, NamedTuple, List, Tuple, Dict, Iterable
import re
import math
from collections import defaultdict


def tokenize(text: str) -> Set[str]:
    text = text.lower()  # конвертация в нижний регистр
    all_words = re.findall("[a-z']+", text)  # извлечение слов и
    return set(all_words)  # удаление повторов.


assert tokenize("Data Science is science") == {"data", "science", "is"}


# определим тип тренировочных данных
class Message(NamedTuple):
    text: str
    is_spam: bool


class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # сглаживающий фактор

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message]) -> None:
        for message in messages:
            # увеличить кол-во сообщений
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1

            # увеличить кол-во появлений слов
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        """ возвращает P(лексема | спам) или P(лексема | неспам)"""
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        # перебрать все слова в лексиконе
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)
            # если *лексема* появляется в сообщении, то добавить вероятность ее встретить
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            # Иначе добавить лог. вероятность ее НЕ встретить: log(1 - вероятность встретить)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_ham + prob_if_spam)



# ______ 2. Тестирование модели_______

messages = [Message("spam rules", is_spam=True),
            Message("ham rules", is_spam=False),
            Message("hello ham", is_spam=False)]

model = NaiveBayesClassifier(k=0.5)
model.train(messages)

assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

text = "hello spam"

probs_if_spam = [
    (1 + 0.5) / (1 + 2 * 0.5),  # "spam" (присутствует)
    1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham" (не присутствует)
    1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (не присутствует)
    (0 + 0.5) / (1 + 2 * 0.5)  # "hello"(присутствует)
]

probs_if_ham = [
    (0 + 0.5) / (2 + 2 * 0.5),  # "spam" (присутствует)
    1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham" (не присутствует)
    1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (не присутствует)
    (1 + 0.5) / (2 + 2 * 0.5)  # "hello"(присутствует)
]

p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

# примерно 0.83
assert model.predict(text) == p_if_spam / (p_if_ham + p_if_spam)


# ______3. Применение Модели_____

# 3.1 Скачивание и распаковка набора данных

from io import BytesIO  # необходимо трактовать байты как файл
import requests  # для скачивания файлов, которые
import tarfile  # находятся в формате .tar.bz

BASE_URl = 'https://spamassassin.apache.org/old/publiccorpus'
FILES = ["20021010_easy_ham.tar.bz2",
         "20021010_hard_ham.tar.bz2",
         "20021010_spam.tar.bz2"]

OUTPUT_DIR = 'spam_data'

for filename in FILES:
    # получение содержимого файлов
    # в каждом URL
    content = requests.get(f"{BASE_URl}/{filename}").content

    # обертка байтов в памяти, для использования их как "файл"
    fin = BytesIO(content)

    # извлечь файлы в указанный выходной каталог
    with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
        tf.extractall(OUTPUT_DIR)

# 3.2 Просмотр тематической строки каждого письма

import glob

path = 'spam_data/*/*'

data: List[Message] = []
for filename in glob.glob(path):
    is_spam = "ham" not in filename

    with open(filename, errors='ignore') as email_file:
        for line in email_file:
            if line.startswith("Subject:"):
                subject = line.lstrip("Subject:")
                data.append(Message(subject, is_spam))
                break  # конец работы с этим файлом

# 3.3 Разбиение данных на тренировочные и тестовые

import random
from machine_learning import split_data

random.seed(0)
train_messages, test_messages = split_data(data, 0.75)
model = NaiveBayesClassifier()
model.train(train_messages)

# 3.4 генерация нескольких предсказаний и проверка работоспособности модели
from collections import Counter

predictions = [(message, model.predict(message.text))
               for message in test_messages]

confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                           for message, spam_probability in predictions)
print(confusion_matrix)


def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
    prob_if_spam, prob_if_ham = model._probabilities(token)

    return prob_if_spam / (prob_if_spam + prob_if_ham)


words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))

print("наиболее_спамные_слова", words[-10])
print("наименее_спамные_слова", words[:10])


# простой стеммер
def drop_final_s(word):
    return re.sub("s$", "", word)
