#_______Энтропия________
import random
from typing import List, Any, Dict, TypeVar, NamedTuple, Union
import math
from collections import Counter, defaultdict


def entropy(class_probabilities: List[float]) -> float:
    """ с учетом списка классовых вероятностей вычислить энтропию"""
    return sum(-p * math.log(p, 2)
               for p in class_probabilities
               if p > 0)  # игнорировать нулевые вероятности


assert entropy([1.0]) == 0
assert entropy([0.5, 0.5]) == 1
assert 0.81 < entropy([0.25, 0.75]) < 0.82


def class_probabilities(labels: List[Any]) -> List[float]:
    total_count = len(labels)
    return [count / total_count
            for count in Counter(labels).values()]


def data_entropy(labels:List[Any]) -> float:
    return entropy(class_probabilities(labels))

assert data_entropy(['a']) == 0
assert data_entropy([True, False]) == 1
assert data_entropy([3, 4, 4, 4]) == entropy([0.25, 0.75])


def partition_entropy(subsets: List[List[Any]]) -> float:
    """возвращает энтропию из этого подразделения данных на подмнржества"""
    total_count = sum(len(subset) for subset in subsets)

    return sum(data_entropy(subset) * len(subset) / total_count
               for subset in subsets)

#________СОЗДАНИЕ ЛЕРЕВА РЕШЕНИЙ, алгоритм ID3________

from typing import NamedTuple, Optional

class Candidate(NamedTuple):
    level: str    # уровень
    lang: str    # язык
    tweets: bool    # twitter
    phd: bool    # степень
    did_well: Optional[bool] = None    # позволить непомеченные данные

inputs = [
     ({'level':'Senior','lang':'Java','tweets':'no','phd':'no'},    False),
     ({'level':'Senior','lang':'Java','tweets':'no','phd':'yes'},   False),
     ({'level':'Mid','lang':'Python','tweets':'no','phd':'no'},     True),
     ({'level':'Junior','lang':'Python','tweets':'no','phd':'no'},  True),
     ({'level':'Junior','lang':'R','tweets':'yes','phd':'no'},      True),
     ({'level':'Junior','lang':'R','tweets':'yes','phd':'yes'},    False),
     ({'level':'Mid','lang':'R','tweets':'yes','phd':'yes'},        True),
    ({'level':'Senior','lang':'Python','tweets':'no','phd':'no'}, False),
    ({'level':'Senior','lang':'R','tweets':'yes','phd':'no'},      True),
    ({'level':'Junior','lang':'Python','tweets':'yes','phd':'no'}, True),
    ({'level':'Senior','lang':'Python','tweets':'yes','phd':'yes'},True),
    ({'level':'Mid','lang':'Python','tweets':'no','phd':'yes'},    True),
    ({'level':'Mid','lang':'Java','tweets':'yes','phd':'no'},      True),
    ({'level':'Junior','lang':'Python','tweets':'no','phd':'yes'},False)
]

T = TypeVar('T')    # обобщенный тип для входов


def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
    """подразделять входы на списки на основе заданного атрибута"""
    partitions: Dict[Any, List[T]] = defaultdict(list)
    for input in inputs:
        key = getattr(input, attribute)    # значение заданного атрибута
        partitions[key].append(input)    # добавить вход в правильное подразделение

    return partitions


def partition_entropy_by(inputs: List[Any],
                         attribute: str,
                         label_attribute: str) -> float:
    """вычислить энтропию, соответствующую заданному подразделу"""
    # подразделы состоят из входов
    partitions = partition_by(inputs, attribute)
    # но энтропия partition_entropy нуждается только в классовых метках
    labels = [[getattr(input, label_attribute) for input in partition]
              for partition in partitions.values()]

    return partition_entropy(labels)

# поиск подразделения с минимальной энтропией для всего набора данных
for key in ['level', 'lang', 'tweets', 'phd']:
    print(key, partition_entropy_by(inputs, key, 'did_well'))

senior_inputs = [input for input in input.level == 'Senior']

class Leaf(NamedTuple):
    value: Any

class Split(NamedTuple):
    attribute: str
    subtrees: dict
    dafault_value: Any = None

DicisionTree = Union[Leaf, Split]

hiring_tree = Split('level', {    # Рассмотреть уровень "level"
    'Junior': Split('phd', {      # Если уровень равен "Junior", обратиться
                                  # к "phd"
        False: Leaf(True),        # Если "phd" равен False, предсказать True
        True: Leaf(False)         # Если "phd" равен True, предсказать False
    }),
    'Mid': Leaf(True),            # Если "level" равен "Mid",
                                  # просто предсказать True
    'Senior': Split('tweets', {   # если "level" равен "Senior", обратиться
                                  # к "tweets"
        False: Leaf(False),       # если "tweets равен False,
                                  # предсказать False
        True: Leaf(True)          # если "tweets равен True, предсказать True
    })
})


def classify(tree: DicisionTree, input: Any) -> Any:
    """классифицировать вход, используя заданное дерево решений"""
    # если это листовой узел, то вернуть его значение
    if isinstance(tree, Leaf):
        return tree.value

    # в противном случае это дерево состоит из атрибута,
    # по которому проводится разбиение,
    # и словаря, ключи которого являются значениями этого атрибута
    # и значениями которого являются поддеревьями, рассмотренными далее
    subtree_key = getattr(input, tree.attribute)

    if subtree_key not in tree.subtrees:    # если для ключа нет поддерева, то
        return tree.dafault_value           # вернуть значение по умолчанию

    subtree = tree.subtrees[subtree_key]    # выбрать соответствующее поддерево

    return classify(subtree, input)         # и использовать для классификации входа


def build_tree_id3(inputs: List[Any],
                   split_attributes: List[str],
                   target_attribute: str) -> DicisionTree:

    # подсчитать целевые метки
    lebel_counts = Counter(getattr(input, target_attribute)
                           for input in inputs)
    most_common_label = lebel_counts.most_common(1)[0][0]

    # если имеется уникальная метка, то предсказать ее
    if len(lebel_counts) == 1:
        return Leaf(most_common_label)


    # если больше не осталось атрибутов, по которым проводить
    # разбиение, то вернуть наиболее распространенную метку
    if not split_attributes:
        return Leaf(most_common_label)

    # в противном случае разбить по наилучшему атрибуту
    def split_entropy(atribute: str) -> float:
        """ Вспомогательная функция для отыскания наилучшего атрибута"""
        return partition_entropy_by(inputs, atribute, target_attribute)

    best_attribute = min(split_attributes, key=split_entropy)

    partitions = partition_by(inputs, best_attribute)
    new_attributes = [a for a in split_attributes if a != best_attribute]

    # рекурсивно строить поддеревья
    subtrees = {attribute_value: build_tree_id3(subset,
                                                 new_attributes,
                                                 target_attribute)
                for attribute_value, subset in partitions.items()}

    return Split(best_attribute, subtrees,
                 dafault_value=most_common_label)



# Случайные леса. Ансамблевое обучение(пример)
# если уже осталось мало кандидатов на разбиение, то обратиться ко всем
if len(split_candidates) <= self.num_split_coandidates:
    sampled_split_candidates = split_candidates
    # в противном случае подобрать случайный образец
else:
    sampled_split_candidates = random.sample(split_candidates,
                                             self.num_split_coandidates)

# теперь наилучший атрибут только из этих кандидатов
best_attribute = min(sampled_split_candidates, key=split_entropy)
partitions = partition_by(inputs, best_attribute)

