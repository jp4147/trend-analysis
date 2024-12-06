from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import numpy as np


from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import numpy as np

class DataSplit:
    def __init__(self, data, normalize=False, batch=16):
        """
        A class to split embedding data into train, validation, and test sets,
        and create corresponding DataLoaders.

        :param data: List of dictionaries where each entry contains 'emb', 'label', and optionally 'age'.
        :param normalize: Whether to normalize embeddings to unit norm.
        :param batch: Batch size for the DataLoader.
        """
        self.batch = batch

        # Validate data structure
        if not isinstance(data, list) or not all('emb' in item and 'label' in item for item in data):
            raise ValueError("Data must be a list of dictionaries with 'emb' and 'label' keys.")

        # Normalize embeddings if required
        self.data = [
            {
                'emb': self.normalize_emb(item['emb']) if normalize else item['emb'],
                'label': item['label'],
                'age': item.get('age', None),
            }
            for item in data
        ]

        # Split data indices into train, validation, and test sets
        self.ids_train, self.ids_val, self.ids_test = self.split_ids()

        # Create datasets for each split
        self.train_data = self.create_data_list(self.ids_train)
        self.val_data = self.create_data_list(self.ids_val)
        self.test_data = self.create_data_list(self.ids_test)

        # Create DataLoaders for each split
        self.train_loader = DataLoader(self.train_data, batch_size=batch, shuffle=True, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(self.val_data, batch_size=batch, shuffle=False, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(self.test_data, batch_size=batch, shuffle=False, collate_fn=self.collate_fn)

    def normalize_emb(self, emb):
        """
        Normalize embedding to unit norm.
        """
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def split_ids(self):
        """
        Split data indices into train, validation, and test sets.
        """
        indices = list(range(len(self.data)))
        labels = [item['label'] for item in self.data]

        # Perform splits
        ids_train, ids_test, labels_train, labels_test = train_test_split(
            indices, labels, test_size=0.2, stratify=labels, random_state=42
        )
        ids_train, ids_val, labels_train, labels_val = train_test_split(
            ids_train, labels_train, test_size=0.25, stratify=labels_train, random_state=42
        )
        return ids_train, ids_val, ids_test

    def create_data_list(self, ids_subset):
        """
        Create a list of tuples (embedding, label, age) for a given subset of indices.
        """
        return [
            (
                torch.tensor(self.data[idx]['emb'], dtype=torch.float32),
                self.data[idx]['label'],
                self.data[idx].get('age', None),
            )
            for idx in ids_subset
        ]

    def collate_fn(self, batch):
        """
        Collate function for DataLoader to batch embeddings, labels.
        """
        embeddings = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
        return embeddings, labels
