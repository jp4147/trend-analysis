from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn

class DataSplit:
    def __init__(self, data, normalize = True, batch = 16):
        self.batch = batch
        
        if normalize:
            print('normalize data')
            all_measurements = [item for seq_label in data.values() for item in seq_label['seq'] if not np.isnan(item)]
            all_measurements = np.array(all_measurements)
            data_min = all_measurements.min()
            data_max = all_measurements.max()
            
            desired_min = 0.1
            desired_max = 1.0
            
            self.data = {
                id: {'seq': self.normalize_seq(seq_label['seq'], data_min, data_max, desired_min, desired_max), 
                    'label':seq_label['label']}
                for id, seq_label in data.items()
                }
            
        else:
            self.data = data

        self.max_len = max([len(seq_label['seq']) for seq_label in self.data.values()])
        self.ids_train, self.ids_val, self.ids_test = self.split_ids()
        
        print('create datasets')
        self.train_data = self.create_data_list(self.ids_train)
        self.val_data = self.create_data_list(self.ids_val)
        self.test_data = self.create_data_list(self.ids_test)
        
        print('create trainloaders')
        self.train_loader = DataLoader(self.train_data, batch_size=batch, shuffle=True, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(self.val_data, batch_size=batch, shuffle=True, collate_fn=self.collate_fn)
        self.test_loader = DataLoader(self.test_data, batch_size=batch, shuffle=False, collate_fn=self.collate_fn)
        
    def normalize_seq(self, seq, data_min, data_max, desired_min, desired_max):
        normalized_seq = ((seq - data_min) / (data_max - data_min)) * (desired_max - desired_min) + desired_min
        normalized_seq = np.nan_to_num(normalized_seq, nan=-0.001)  # Replace np.nan with -0.001
        return normalized_seq
    
    def split_ids(self):
        ids, y = [], []
        for k, v in self.data.items():
            ids.append(k)
            y.append(v['label'])
            
        # stratify_labels = [''.join(map(str, label)) for label in y]
        ids_train, ids_test, y_train, y_test = train_test_split(ids, y, test_size=0.2, stratify=y, random_state=42)
        # stratify_labels_train = [''.join(map(str, label)) for label in y_train]
        ids_train, ids_val, y_train, y_val = train_test_split(ids_train, y_train, test_size=0.25, stratify=y_train, random_state=42)  # 0.25 x 0.8 = 0.2
        return ids_train, ids_val, ids_test
        
    def collate_fn(self, batch):
        # seq=  pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
            
        fixed_length = self.max_len
        seq = pad_sequence(
            [torch.cat([item[0], torch.zeros(fixed_length - item[0].size(0), item[0].size(1))]) if item[0].size(0) < fixed_length else item[0] for item in batch],
            batch_first=True, padding_value=0
        )
        
        # age=  pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
        label = torch.tensor([item[1] for item in batch])

        return seq, label

    def create_data_list(self, ids_subset):
        data_list = []
        for key in ids_subset:
            seq = torch.tensor(self.data[key]['seq'], dtype = torch.float32).unsqueeze(-1)
            # age = torch.tensor(self.data[key]['age'])
            la = self.data[key]['label']
            # data_list.append((seq, age, la))
            data_list.append((seq, la))
        return data_list
