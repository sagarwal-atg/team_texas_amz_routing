from models.models import ARC_Classifier
from dataloaders.irl_dataset import IRLDataset, LinkFeatures, RouteFeatures
from torch.utils.data import DataLoader

null_callback = lambda *args, **kwargs: None

BATCHSIZE = 32

class Path:
    base = '/home/josiah/code/arc/my-app/data/model_build_inputs/small/'
    route = base + 'route_data.json'
    labels = base + 'actual_sequences.json'
    travel_times = base + 'travel_times.json'
    packages = base + 'package_data.json'


def fit(model, dataloader, epochs=1, verbose=0,
        cb_after_batch_update=null_callback, cb_after_epoch=null_callback):

    for epoch in range(epochs):  # loop over the dataset multiple times
        epoch_loss = 0
        for data in dataloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            loss = model.train_on_batch(inputs, labels)
            epoch_loss += loss
            cb_after_batch_update(loss)
        cb_after_epoch(epoch, epoch_loss)
        if verbose > 0:
            print(f'Epoch: {epoch}, Loss {epoch_loss:.4f}')

def main():
    paths = Path()
    dataset = IRLDataset(paths.route, paths.labels, paths.travel_times, paths.packages)
    trainloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True)
    print('Loaded Data')

    model = ARC_Classifier(
        dataset.max_route_len,
        dataset.num_features,
        hidden_sizes=[256]
    )

    fit(model, trainloader, epochs=100, verbose=1)
    print('Finished Training')

main()