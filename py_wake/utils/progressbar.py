
def progressbar(it, show=True):
    if show and len(it) > 1:
        try:
            from tqdm import tqdm
            return tqdm(it)
        except Exception:
            return it
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
