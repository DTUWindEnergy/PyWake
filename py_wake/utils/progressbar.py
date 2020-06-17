from tqdm import tqdm


def progressbar(it):
    if len(it) > 1:
        return tqdm(it)
    else:
        return it


def main():
    if __name__ == '__main__':
        import time
        for _ in progressbar(range(5)):
            time.sleep(0.1)

        for _ in progressbar(range(1)):
            time.sleep(0.1)


main()
