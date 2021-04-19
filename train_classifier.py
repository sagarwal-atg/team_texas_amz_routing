import torch
from models.models import ARC_Classifier
from dataloaders.irl_dataset import IRLDataset
from torch.utils.data import DataLoader, random_split
import numpy as np

null_callback = lambda *args, **kwargs: None

BATCHSIZE = 32

class Path:
    base = '/home/josiah/code/arc/my-app/data/model_build_inputs/'
    route = base + 'route_data.json'
    sequence = base + 'actual_sequences.json'
    travel_time = base + 'travel_times.json'
    packages = base + 'package_data.json'


def fit(model, dataloader, epochs=1, verbose=0,
        cb_after_batch_update=null_callback, cb_after_epoch=null_callback):

    for epoch in range(epochs):  # loop over the dataset multiple times
        epoch_loss = []
        for data in dataloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            loss = model.train_on_batch(inputs, labels)
            epoch_loss.append(loss.detach())
            cb_after_batch_update(loss)
        cb_after_epoch(epoch, model)
        if verbose > 0:
            accuracy = (model(inputs).argmax(1) == labels).float().mean().item()
            print(f'Epoch: {epoch}, Loss {np.mean(epoch_loss):.4f}, Accuracy: {accuracy:.2f}')


def main():
    paths = Path()
    data = IRLDataset(paths, slice_end=800)
    train_size = int(len(data)*.7)
    test_size = len(data) - train_size
    train, test = random_split(data, [train_size, test_size])
    print(f'Train size: {len(train)}, Test size: {len(test)}')
    train_loader = DataLoader(train, batch_size=BATCHSIZE, shuffle=True)

    def test_cb(epoch, model):
        inputs, labels = test[:]
        with torch.no_grad():
            outputs = model(inputs)
        accuracy = (model(inputs).argmax(1) == labels).float().mean().item()
        loss = model.get_loss(outputs, labels)
        print(f'Epoch: {epoch}, Test Loss {loss:.4f}, Test Accuracy: {accuracy:.2f}')

    model = ARC_Classifier(
        data.max_route_len,
        data.num_features,
        hidden_sizes=[256],
        lr=0.01
    )

    fit(model, train_loader, epochs=500, verbose=1, cb_after_epoch=test_cb)
    print('Finished Training')

main()