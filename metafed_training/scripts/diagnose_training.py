import torch
from models import CNN
from utils import load_pamap2_federations, create_data_loaders, train_client, evaluate
from config import *


def diagnose():
    device = torch.device('cpu')
    federations = load_pamap2_federations('./PAMAP2_Dataset/Protocol', SEQUENCE_LENGTH, NUM_FEDERATIONS)
    train_loaders, test_loaders = create_data_loaders(federations, BATCH_SIZE)

    model = CNN(INPUT_SIZE, NUM_CLASSES).to(device)

    for i in range(len(train_loaders)):
        print(f"\nClient {i+1} before training:")
        # show a batch
        for data, target in train_loaders[i]:
            print('batch data shape', data.shape)
            print('target unique', sorted(set(target.tolist())))
            out = model(data)
            _, pred = out.max(1)
            print('pred unique (before)', sorted(set(pred.tolist())))
            break

        print('Eval acc (before):', evaluate(model, test_loaders[i], device))

        # single-step training debug
        model_copy = train_client(model, train_loaders[i], device, epochs=1, lr=LEARNING_RATE)
        # check predictions after 1 epoch
        for data, target in train_loaders[i]:
            out = model_copy(data)
            _, pred = out.max(1)
            print('pred unique (after 1 epoch)', sorted(set(pred.tolist())))
            break

        print('Eval acc (after local train):', evaluate(model_copy, test_loaders[i], device))


if __name__ == '__main__':
    diagnose()
