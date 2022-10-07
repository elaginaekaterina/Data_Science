from collections import Counter
import csv

with open('email_addresses.txt', 'w') as f:
    f.write("kateelagina@gmail.com\n")
    f.write("kate@m.datasciencester.com\n")
    f.write("kateelagina@m.datasciencester.com\n")


def get_domain(email_address: str) -> str:
    """разбить по '@' и вернуть остаток строки"""
    return email_address.lower().split('@')[-1]


# проверки
assert get_domain('kateelagina@gmail.com') == 'gmail.com'
assert get_domain('kate@m.datasciencester.com') == 'm.datasciencester.com'

with open('email_addresses.txt', 'r') as f:
    domain_counts = Counter(get_domain(line.strip())
                            for line in f
                            if "@" in line)

# разделитель табуляция, нет заголовков
with open('tab_delimited_stock_prices.txt', 'w') as f:
    f.write("""6/20/2014\tAAPL\t90.91
6/20/2014\tMSFT\t41.68
6/20/2014\tFB\t64.5
6/19/2014\tAAPL\t91.86
6/19/2014\tMSFT\t41.51
6/19/2014\tFB\t64.34
""")


def process(date: str, symbol: str, closing_price: float) -> None:
    # Imaginge that this function actually does something.
    assert closing_price > 0.0


with open('tab_delimited_stock_prices.txt') as f:
    tab_reader = csv.reader(f, delimiter='\t')
    for row in tab_reader:
        date = row[0]
        symbol = row[1]
        closing_price = float(row[2])
        process(date, symbol, closing_price)

# делитель :, есть заголовки
with open('colon_delimited_stock_prices.txt', 'w') as f:
    f.write("""date:symbol:closing_price
6/20/2014:AAPL:90.91
6/20/2014:MSFT:41.68
6/20/2014:FB:64.5
""")

with open('colon_delimited_stock_prices.txt') as f:
    colon_reader = csv.DictReader(f, delimiter=':')
    for dict_row in colon_reader:
        date = dict_row["date"]
        symbol = dict_row["symbol"]
        closing_price = float(dict_row["closing_price"])
        process(date, symbol, closing_price)

# Запись с разделителями в файл
todays_prices = {'AAPL': 90.91, 'MSFT': 41.68, 'FB': 64.5}

with open('comma_delimited_stock_prices.txt', 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    for stock, price in todays_prices.items():
        csv_writer.writerow([stock, price])

# попытка обработать самостоятельно
results = [["test1", "success", "Monday"],
           ["test2", "success, kind of", "Tuesday"],
           ["test3", "failure, kind of", "Wednesday"],
           ["test4", "failure, utter", "Thursday"]]

# НЕ ДЕЛАТЬ ТАК!!!
with open('bad_csv.txt', 'w') as f:
    for row in results:
        f.write(",".join(map(str, row)))  # внутри может быть много запятых
        f.write("\n")  # строка может также содержать символ новой строки

# _______HTML и его разбор____________
from bs4 import BeautifulSoup
import requests

url = ("https://github.com/elaginaekaterina/DataScience/blob/main/getting-data.html")
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

first_paragraph = soup.find('p')  # первый тег, можно просто soup.p

first_paragraph_text = soup.p.text  # текст первого элемента <p>
first_paragraph_words = soup.p.text.split()  # слова первого элемента

assert first_paragraph_words == ['This', 'is', 'the', 'first', 'paragraph.']

first_paragraph_id = soup.p['id']  # вызывает KeyError если 'id' отсутствует
first_paragraph_id2 = soup.p.get('id')  # возвращает None если 'id' отсутствует

assert first_paragraph_id == first_paragraph_id2 == 'p1'

# несколько тегов сразу
all_paragraphs = soup.find_all('p')  # или просто soup('p')
paragraphs_with_ids = [p for p in soup('p') if p.get('id')]

assert len(all_paragraphs) == 2
assert len(paragraphs_with_ids) == 1

# поиск тегов с конкретным классом в стилевой таблицы
important_paragraphs = soup('p', {'class': 'important'})
important_paragraphs2 = soup('p', 'important')
important_paragraphs3 = [p for p in soup('p')
                         if 'important' in p.get('class', [])]

assert important_paragraphs == important_paragraphs2 == important_paragraphs3
assert len(important_paragraphs) == 1

# поиск каждого элемента span внутри div
spans_inside_divs = [span
                     for div in soup('div')  # для каждого <div> на странице
                     for span in div('span')]  # отыскать каждый <span> внутри него

assert len(spans_inside_divs) == 3


# ___________Пример______________
# проверяет, упоманиет ли страница какой-либо заданный термин
def paragraph_mentions(text: str, keyword: str) -> bool:
    """
    возвращает True, если <p> внутри упоминаний {keyword} в тексте
    """
    soup = BeautifulSoup(text, 'html5lib')
    paragraphs = [p.get_text() for p in soup('p')]

    return any(keyword.lower() in paragraph.lower()
               for paragraph in paragraphs)


text = """<body><h1>Facebook</h1><p>Twitter</p>"""
assert paragraph_mentions(text, "twitter")  # внутри <p>
assert not paragraph_mentions(text, "facebook")  # не внутри <p>


def main():
    from bs4 import BeautifulSoup
    import requests

    url = "https://www.house.gov/representatives"
    text = requests.get(url).text
    soup = BeautifulSoup(text, "html5lib")

    all_urls = [a['href']
                for a in soup('a')
                if a.has_attr('href')]

    print(len(all_urls))  # 965 много

    import re

    # должны начинаться с http:// or https://
    # должны заканчиваться на .house.gov or .house.gov/
    regex = r"^https?://.*\.house\.gov/?$"

    # тестирование
    assert re.match(regex, "http://joel.house.gov")
    assert re.match(regex, "https://joel.house.gov")
    assert re.match(regex, "http://joel.house.gov/")
    assert re.match(regex, "https://joel.house.gov/")
    assert not re.match(regex, "joel.house.gov")
    assert not re.match(regex, "http://joel.house.com")
    assert not re.match(regex, "https://joel.house.gov/biography")

    # применим
    good_urls = [url for url in all_urls if re.match(regex, url)]

    print(len(good_urls))  # 862

    num_original_good_urls = len(good_urls)

    good_urls = list(set(good_urls))

    print(len(good_urls))  # 431

    assert len(good_urls) < num_original_good_urls

    html = requests.get('https://jayapal.house.gov').text
    soup = BeautifulSoup(html, 'html5lib')

    # используем множество set, т.к ссылки могут появляться многократно
    links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}

    print(links)  # {'/media/press-releases'}

    from typing import Dict, Set

    press_releases: Dict[str, Set[str]] = {}

    for house_url in good_urls:
        html = requests.get(house_url).text
        soup = BeautifulSoup(html, 'html5lib')
        pr_links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}
        print(f"{house_url}: {pr_links}")
        press_releases[house_url] = pr_links

    for house_url, pr_links in press_releases.items():
        for pr_link in pr_links:
            url = f"{house_url}/{pr_link}"
            text = requests.get(url).text

            if paragraph_mentions(text, 'data'):
                print(f"{house_url}")
                break  # с этим адресом house_url закончено

    # ___________________Использование интерфейсов API__________________
    # JSON and XML
    {"title": "Data Science Book",
     "author": "Joel Grus",
     "publicationYear": 2019,
     "topics": ["data", "science", "data science"]}

    import json

    serialized = """{ "title" : "Data Science Book",
                      "author" : "Joel Grus",
                      "publicationYear" : 2019,
                      "topics" : [ "data", "science", "data science"] }"""

    # разобрать JSON создав Python'овский словарь
    deserialized = json.loads(serialized)
    assert deserialized["publicationYear"] == 2019
    assert "data science" in deserialized["topics"]

    import requests, json

    github_user = "elaginaekaterina"
    endpoint = f"https://api.github.com/users/{github_user}/repos"

    repos = json.loads(requests.get(endpoint).text)

    from collections import Counter
    from dateutil.parser import parse

    dates = [parse(repo["created_at"]) for repo in repos]
    month_counts = Counter(date.month for date in dates)
    weekday_counts = Counter(date.weekday() for date in dates)

    last_5_repositories = sorted(repos,
                                 key=lambda r: r["pushed_at"],
                                 reverse=True)[:5]

    last_5_languages = [repo["language"]
                        for repo in last_5_repositories]

    # ________Пример. Twitter__________
    import os

    # ключ и секрет в коде
    CONSUMER_KEY = os.environ.get("TWITTER_CONSUMER_KEY")
    CONSUMER_SECRET = os.environ.get("TWITTER_CONSUMER_SECRET")

    import webbrowser
    from twython import Twython

    # получить временного клиента для извлечения аутентификационного url
    temp_client = Twython(CONSUMER_KEY, CONSUMER_SECRET)
    temp_creds = temp_client.get_authentication_tokens()
    url = temp_creds['auth_url']

    # посетить URL для авторизации приложения и получения PIN
    print(f"перейдите на {url} и получите PIN-коде и вставьте его ниже")
    webbrowser.open(url)
    PIN_CODE = input("пожалуйста введите PIN: ")

    # используем PIN_CODE для получения фактических токенов
    auth_client = Twython(CONSUMER_KEY,
                          CONSUMER_SECRET,
                          temp_creds['oauth_token'],
                          temp_creds['oauth_token_secret'])
    final_step = auth_client.get_authorized_tokens(PIN_CODE)
    ACCESS_TOKEN = final_step['oauth_token']
    ACCESS_TOKEN_SECRET = final_step['oauth_token_secret']

    # получим новый экземпляр Twython, используя их
    twitter = Twython(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    from twython import TwythonStreamer

    # Добавление данных в глобальную переменную - довольно плохой вариант
    # но это делает пример намного проще
    tweets = []

    class MyStreamer(TwythonStreamer):
        def on_success(self, data):
            """
            Что мы делаем, когда твиттер отправляет нам данные?
            Здесь данные будут представлять собой словарь Python, представляющий твит.
            """
            # Мы хотим собирать твиты только на английском языке
            if data.get('lang') == 'en':
                tweets.append(data)
                print(f"received tweet #{len(tweets)}")

            # остановиться, когда достаточно
            if len(tweets) >= 100:
                self.disconnect()

        def on_error(self, status_code, data):
            print(status_code, data)
            self.disconnect()

    stream = MyStreamer(CONSUMER_KEY, CONSUMER_SECRET,
                        ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    # начинает потреблять общедоступные статусы, содержащие ключевое слово'data'
    stream.statuses.filter(track='data')

    # если бы вместо этого мы хотели начать потреблять выборку *всех* публичных статусов
    # stream.statuses.sample()


if __name__ == "__main__": main()
