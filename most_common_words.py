import sys
from collections import Counter

if __name__ == "__main__":
    # передать число слов в качестве первого аргумента
    try:
        num_words = int(sys.argv[1])
    except:
        print("Применение: most_common_words.py num_words")
        sys.exit(1)  # ненулевой код выхода говорит об ошибке

    counter = Counter(word.lower()
                      for line in sys.stdin
                      for word in line.strip().split()  # разбить строку по пробелам

                      if word)  # пропустить 'пустые' слова

    for word, count in counter.most_common(num_words):
        sys.stdout.write(str(count))
        sys.stdout.write("\t")
        sys.stdout.write(word)
        sys.stdout.write("\n")
