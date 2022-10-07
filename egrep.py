import sys, re

if __name__ == "__main__":

    # sys.argv список аргументов командной строки
    # sys.argv[0] имя самой программы
    # sys.argv[1] регулярное выражение, указываемое в командной строке
    regex = sys.argv[1]

    # for every line passed into the script
    for line in sys.stdin:
        # if it matches the regex, write it to stdout
        if re.search(regex, line):
            sys.stdout.write(line)