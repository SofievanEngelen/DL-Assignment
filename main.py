# This is a sample Python script.
from preprocessing import preprocess_dir

datadir = './data/small-train'


def main():
    print(preprocess_dir(datadir))


if __name__ == '__main__':
    main()
