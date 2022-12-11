from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

DATA_PATH = "./train/data/"


def get_mutag_dataset():
    return TUDataset(root=DATA_PATH, name='MUTAG')


def graph_stat(dataset):
    """
    TODO: calculate the statistics of the ENZYMES dataset.

    Outputs:
        min_num_nodes: min number of nodes
        max_num_nodes: max number of nodes
        mean_num_nodes: average number of nodes
        min_num_edges: min number of edges
        max_num_edges: max number of edges
        mean_num_edges: average number of edges
    """

    # your code here
    min_num_nodes = 1e3
    max_num_nodes = -1
    mean_num_nodes = 0

    min_num_edges = 1e3
    max_num_edges = -1
    mean_num_edges = 0
    for data in dataset:
        n, m = data.num_nodes, data.num_edges
        min_num_nodes = min(min_num_nodes, n)
        max_num_nodes = max(max_num_nodes, n)
        mean_num_nodes += n

        min_num_edges = min(min_num_edges, m)
        max_num_edges = max(max_num_edges, m)
        mean_num_edges += m
    return (
        min_num_nodes, max_num_nodes, mean_num_nodes /
        len(dataset), min_num_edges, max_num_edges, mean_num_edges/len(dataset)
    )


def print_data_stats(dataset: TUDataset):
    data = dataset[0]
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Number of features: {data.num_node_features}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


def get_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_test_split(dataset: TUDataset, rate=0.8):
    # shuffle
    dataset = dataset.shuffle()
    # split
    split_idx = int(len(dataset) * rate)
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    print("len train:", len(train_dataset))
    print("len test:", len(test_dataset))
    return train_dataset, test_dataset