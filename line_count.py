import sys

if __name__ == "__main__":

    count = 0
    for line in sys.stdin:
        count += 1

    # печать выводится в консоль sys.stdout
    print(count)