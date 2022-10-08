from typing import Tuple, Sequence, List, Any, Callable, Dict, Iterator
from collections import defaultdict

# несколько псевдонимов типов, которые будут использоваться позже
Row = Dict[str, Any]  # строка БД
WhereClause = Callable[[Row], bool]  # предикат для единственной строки
HavingClause = Callable[[List[Row]], bool]  # предикат над многочисленными строками


class Table:
    def __init__(self, columns: List[str], types: List[type]) -> None:
        assert len(columns) == len(types)
        self.columns = columns
        self.types = types
        self.rows: List[Row] = []  # данных пока нет

    # вспомогательный метод для получения типа столбца
    def col2type(self, col: str) -> type:
        idx = self.columns.index(col)  # найти индекс столбца
        return self.types[idx]  # и вернуть его тип

    def insert(self, values: list) -> None:
        # проверить правильность числа значений
        if len(values) != len(self.types):
            raise ValueError(f"Требуется{len(self.types)} значений")

        # проверить допустимость типов значений
        for value, typ3 in zip(values, self.types):
            if not isinstance(value, typ3) and value is not None:
                raise TypeError(f"Ожидаемый тип {typ3}, но получено {value}")

        # добавить соответствующий словарь как "строку"
        self.rows.append(dict(zip(self.columns, values)))

    def __getitem__(self, idx: int) -> Row:
        return self.rows[idx]

    def __iter__(self) -> Iterator[Row]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self):
        """Структурированное представление таблицы:
        столбцы затем строки"""
        rows = "\n".join(str(row) for row in self.rows)
        return f"{self.columns}\n{rows}"

    def update(self,
               updates: Dict[str, Any],
               predicate: WhereClause = lambda row: True):
        # убедиться, что обновления имеют допустимые имена и типы
        for column, new_value in updates.items():
            if column not in self.columns:
                raise ValueError(f"недопустимый столбец: {column}")

            typ3 = self.col2type(column)
            if not isinstance(new_value, typ3) and new_value is not None:
                raise TypeError(f"ожидаемый тип {typ3}, но получено {new_value}")

        # обновить
        for row in self.rows:
            if predicate(row):
                for column, new_value in updates.items():
                    row[column] = new_value

    def delete(self, predicate: WhereClause = lambda row: True) -> None:
        """Удалить все строки, совпадающие с предикатом"""
        self.rows = [row for row in self.rows if not predicate(row)]

    def select(self,
               keep_columns: List[str] = None,
               additional_columns: Dict[str, Callable] = None) -> 'Table':

        if keep_columns is None:  # если ни один столбец не указан,
            keep_columns = self.columns  # то вернуть все столбцы

        if additional_columns is None:
            additional_columns = {}

        # имена и типы новых столбцов
        new_columns = keep_columns + list(additional_columns.keys())
        keep_types = [self.col2type(col) for col in keep_columns]

        # получить возвращаемый тип из аннотации типа
        add_types = [calculation.__annotations__['return']
                     for calculation in additional_columns.values()]

        # создать новую таблицу для результатов
        new_table = Table(new_columns, keep_types + add_types)

        for row in self.rows:
            new_row = [row[column] for column in keep_columns]
            for column_name, calculation in additional_columns.items():
                new_row.append(calculation(row))
            new_table.insert(new_row)

        return new_table

    def where(self, predicate: WhereClause = lambda row: True) -> 'Table':
        """вернуть только строки, удовлетворяющие переданному предикату"""
        where_table = Table(self.columns, self.types)
        for row in self.rows:
            if predicate(row):
                values = [row[column] for column in self.columns]
                where_table.insert(values)
        return where_table

    def limit(self, num_rows: int) -> 'Table':
        """вернуть только первые 'num_rows' строк"""
        limit_table = Table(self.columns, self.types)
        for i, row in enumerate(self.rows):
            if i >= num_rows:
                break
            values = [row[column] for column in self.columns]
            limit_table.insert(values)
        return limit_table

    def group_by(self,
                 group_by_columns: List[str],
                 aggregates: Dict[str, Callable],
                 having: HavingClause = lambda group: True) -> 'Table':

        grouped_rows = defaultdict(list)

        # заполнить группы
        for row in self.rows:
            key = tuple(row[column] for column in group_by_columns)
            grouped_rows[key].append(row)

        # результирующая таблица состоит из
        # столбцов group_by и агрегатов
        new_columns = group_by_columns + list(aggregates.keys())
        group_by_types = [self.col2type(col)
                          for col in group_by_columns]
        aggregate_types = [agg.__annotations__['return']
                           for agg in aggregates.values()]
        result_table = Table(new_columns, group_by_types + aggregate_types)

        for key, rows in grouped_rows.items():
            if having(rows):
                new_row = list(key)
                for aggregate_name, aggregate_fn in aggregates.items():
                    new_row.append(aggregate_fn(rows))
                result_table.insert(new_row)

        return result_table

    def order_by(self, order: Callable[[Row], Any]) -> 'Table':
        new_table = self.select()
        new_table.rows.sort(key=order)
        return new_table

    def join(self, other_table: 'Table',
             left_join: bool = False) -> 'Table':

        join_on_columns = [c for c in self.columns  # столбцы
                           if c in other_table.columns]  # обеих таблицах

        additional_columns = [c for c in other_table.columns  # столбцы только
                              if c not in join_on_columns]  # правой таблице

        # все столбцы из левой таблицы + дополнительные
        # additional_columns из правой
        new_columns = self.columns + additional_columns
        new_types = self.types + [other_table.col2type(col)
                                  for col in additional_columns]

        join_table = Table(self.columns + additional_columns)

        for row in self.rows:
            def is_join(other_row):
                return all(other_row[c] == row[c]
                           for c in join_on_columns)

            other_rows = other_table.where(is_join).rows

            # каждая строка, удовлетворяющая предикату,
            # производит результирующую строку
            for other_row in other_rows:
                join_table.insert([row[c] for c in self.columns] +
                                  [other_row[c] for c
                                   in additional_columns])

            # если ни одна строка не удовлетворяет условию и это
            # объединение left join, то выделить стоку со значчением None
            if left_join and not other_rows:
                join_table.insert([row[c] for c in self.columns] +
                                  [None for c in additional_columns])

        return join_table


# Create and update

users = Table(['user_id', 'name', 'num_friends'], [int, str, int])
users.insert([0, "Hero", 0])
users.insert([1, "Dunn", 2])
users.insert([2, "Sue", 3])
users.insert([3, "Chi", 3])
users.insert([4, "Thor", 3])
users.insert([5, "Clive", 2])
users.insert([6, "Hicks", 3])
users.insert([7, "Devin", 2])
users.insert([8, "Kate", 2])
users.insert([9, "Klein", 3])
users.insert([10, "Jen", 1])

assert len(users) == 11
assert users[1]['name'] == 'Dunn'
assert users[1]['num_friends'] == 2

users.update({'num_friends': 3},
             lambda row: row['user_id'] == 1)

assert users[1]['num_friends'] == 3

# SELECT

# SELECT * FROM users;
all_users = users.select()
assert len(all_users) == 11

# SELECT * FROM users LIMIT 2;
two_users = users.limit(2)
assert len(two_users) == 2

# SELECT user_id FROM users;
just_ids = users.select(keep_columns=["user_id"])
assert just_ids.columns == ["user_id"]

# SELECT user_id FROM users WERE name = 'Dunn';
dunn_ids = (
    users.where(lambda row: row["name"] == "Dunn").select(keep_columns=["user_id"]))
assert len(dunn_ids) == 1
assert dunn_ids[0] == {"user_id": 1}


# SELECT LENGTH(name) AS name_length FROM users;
def name_length(row) -> int: return len(row["name"])


name_length = users.select(keep_columns=[],
                           additional_columns={"name_length": name_length})
assert name_length[0]['name_length'] == len("Hero")


def min_user_id(rows) -> int:
    return min(row["user_id"] for row in rows)


def length(rows) -> int:
    return len(rows)


# group_by

stats_by_length = users \
    .select(additional_columns={"name_length": name_length}) \
    .group_by(group_by_columns=["name_length"],
              aggregates={"min_user_id": min_user_id,
                          "num_users": length})


def first_letter_of_name(row: Row) -> str:
    return row["name"][0] if row["name"] else ""


def average_num_friends(rows: List[Row]) -> float:
    return sum(row["num_friends"] for row in rows) / len(rows)


def enough_friends(rows: List[Row]) -> bool:
    return average_num_friends(rows) > 1


avg_friends_by_letter = users \
    .select(additional_columns={'first_letter': first_letter_of_name}) \
    .group_by(group_by_columns=['first_letter'],
              aggregates={"avg_num_friends": average_num_friends},
              having=enough_friends)


def sum_user_ids(rows: List[Row]) -> int: return sum(row["user_id"] for row in rows)


user_id_sum = users \
    .where(lambda row: row["user_id"] > 1) \
    .group_by(group_by_columns=[],
              aggregates={"user_id_sum": sum_user_ids})

# Order_by

friendliest_letters = avg_friends_by_letter \
    .order_by(lambda row: -row["avg_num_friends"]) \
    .limit(4)

# JOINs

user_interests = Table(["user_id", "interest"])
user_interests.insert([0, "SQL"])
user_interests.insert([0, "NoSQL"])
user_interests.insert([2, "SQL"])
user_interests.insert([2, "MySQL"])

sql_users = users \
    .join(user_interests) \
    .where(lambda row: row["interest"] == "SQL") \
    .select(keep_columns=["name"])


def count_interests(rows):
    """Подситывает количество строк с ненулевыми интересами"""
    return len([row for row in rows if row["interest"] is not None])


user_interest_counts = users \
    .join(user_interests, left_join=True) \
    .group_by(group_by_columns=["user_id"],
              aggregates={"num_interests": count_interests})

# подзапросы
likes_sql_user_ids = (
    user_interests.where(lambda row: row["interests"] == "SQL").select(keep_columns=["user_id"])
)
likes_sql_user_ids.group_by(group_by_columns=[],
                            aggregates={"min_user_id": min_user_id})
