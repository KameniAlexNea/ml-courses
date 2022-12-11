
from .models import GCN
from .dataset import *
import torch


def train(train_loader):
    gcn.train()
    for data in train_loader:  # Iterate in batches over the training dataset.
        """
        TODO: train the model for one epoch.
        
        Note that you can acess the batch data using `data.x`, `data.edge_index`, `data.batch`, `data,y`.
        """
        
        # your code here
        y_pred = gcn(data.x, data.edge_index, data.batch)
        loss = criterion(y_pred, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(loader):
    gcn.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = gcn(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


if __name__ == '__main__':
    gcn = GCN()

    # optimizer
    optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01)
    # loss
    criterion = torch.nn.CrossEntropyLoss()
    dataset = get_mutag_dataset()
    train_ds, test_ds = train_test_split(dataset, 0.8)
    train_loader = get_dataloader(train_ds, 500, True)
    test_loader = get_dataloader(test_ds, 500, False)
    for epoch in range(200):
        train(train_loader)
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch + 1:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')