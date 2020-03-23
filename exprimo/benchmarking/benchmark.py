import time

from torchvision import transforms
import torchvision
import torch.utils.data
import torch

from exprimo.benchmarking.utils import load_model_with_placement

BATCH_SIZE = 128

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = torchvision.datasets.FakeData(transform=preprocess, size=500)

train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True
)


def train_single_batch(model, data, criterion, optimizer):
    output = model(data[0])
    loss = criterion(output, data[1])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def benchmark_with_placement(model_type, placement='cuda:0', batches=50, drop_batches=1, lr=0.01, verbose=False):
    if verbose:
        print('Starting benchmark...')

    model, criterion, optimizer, input_device, output_device = load_model_with_placement(model_type, placement, lr=lr)

    model.train()
    batch_times = []

    b = 0
    while b < batches + drop_batches:
        for data in train_loader:
            if verbose:
                print(f'Batch {b + 1}/{batches + drop_batches}', end='')

            torch.cuda.synchronize()
            data = data[0].to(input_device), data[1].to(output_device)

            start = time.time()
            train_single_batch(model, data, criterion, optimizer)
            torch.cuda.synchronize()
            end = time.time()

            batch_times.append((end - start) * 1000)

            if verbose:
                print(f' {batch_times[-1]}ms')

            b += 1
            if b >= batches + drop_batches:
                break

    del model, criterion, optimizer, input_device, output_device

    return batch_times[drop_batches:]