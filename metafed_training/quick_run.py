import train


def quick_run():
    # Reduce rounds/epochs for a quick smoke-test run
    train.NUM_ROUNDS = 5
    train.LOCAL_EPOCHS = 1
    train.main()


if __name__ == '__main__':
    quick_run()
