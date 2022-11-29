import torch
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):

    def __init__(self, seqs, hfs):
        self.x = seqs
        self.y = hfs

    def __len__(self):
        """
        TODO: Return the number of samples (i.e. patients).
        """

        # your code here
        return len(self.x)

    def __getitem__(self, index):
        """
        TODO: Generates one sample of data.

        Note that you DO NOT need to covert them to tensor as we will do this later.
        """

        # your code here
        return self.x[index], self.y[index]


def collate_fn(data):
    """
    TODO: Collate the the list of samples into batches. For each patient, you need to pad the diagnosis
        sequences to the sample shape (max # visits, max # diagnosis codes). The padding infomation
        is stored in `mask`.

    Arguments:
        data: a list of samples fetched from `CustomDataset`

    Outputs:
        x: a tensor of shape (# patiens, max # visits, max # diagnosis codes) of type torch.long
        masks: a tensor of shape (# patiens, max # visits, max # diagnosis codes) of type torch.bool
        rev_x: same as x but in reversed time. This will be used in our RNN model for masking 
        rev_masks: same as mask but in reversed time. This will be used in our RNN model for masking
        y: a tensor of shape (# patiens) of type torch.float

    Note that you can obtains the list of diagnosis codes and the list of hf labels
        using: `sequences, labels = zip(*data)`
    """

    sequences, labels = zip(*data)

    y = torch.tensor(labels, dtype=torch.float)

    num_patients = len(sequences)
    num_visits = [len(patient) for patient in sequences]
    num_codes = [len(visit) for patient in sequences for visit in patient]

    max_num_visits = max(num_visits)
    max_num_codes = max(num_codes)

    x = torch.zeros((num_patients, max_num_visits,
                    max_num_codes), dtype=torch.long)
    rev_x = torch.zeros((num_patients, max_num_visits,
                        max_num_codes), dtype=torch.long)
    masks = torch.zeros((num_patients, max_num_visits,
                        max_num_codes), dtype=torch.bool)
    rev_masks = torch.zeros(
        (num_patients, max_num_visits, max_num_codes), dtype=torch.bool)
    for i_patient, patient in enumerate(sequences):
        for j_visit, visit in enumerate(patient):
            """
            TODO: update `x`, `rev_x`, `masks`, and `rev_masks`
            """
            # your code here
            n_visit = len(visit)
            x[i_patient, j_visit, :n_visit] = torch.tensor(
                visit, dtype=torch.long)
            masks[i_patient, j_visit, :n_visit] = 1
            rev_x[i_patient, len(patient) - 1 - j_visit,
                  :n_visit] = torch.tensor(visit, dtype=torch.long)
            rev_masks[i_patient, len(patient) - 1 - j_visit, :n_visit] = 1

    return x, masks, rev_x, rev_masks, y


def load_data(train_dataset, val_dataset, batch_size=32):
    '''
    TODO: Implement this function to return the data loader for  train and validation dataset. 
    Set batchsize to 32. Set `shuffle=True` only for train dataloader.

    Arguments:
        train dataset: train dataset of type `CustomDataset`
        val dataset: validation dataset of type `CustomDataset`
        collate_fn: collate function

    Outputs:
        train_loader, val_loader: train and validation dataloaders

    Note that you need to pass the collate function to the data loader `collate_fn()`.
    '''

    # your code here
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader
