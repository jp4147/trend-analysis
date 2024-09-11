from split_data import DataSplit

from model import*
import torch
import torch.nn as nn
import copy

class Training:
    def __init__(self, data = None, model = None, input_dim=1, hidden_dim=128, output_dim=1, num_heads = 4, num_layers = 2, dropout = 0.1, device: torch.device = torch.device('cpu')):
        self.data = data
        self.device = device 
        
        if len(data[0]['seq'])==1536:
            self.data_splits = DataSplit(data, normalize = False)
            self.train_loader = self.data_splits.train_loader
            self.val_loader = self.data_splits.val_loader
        else:
            self.data_splits = DataSplit(data)
            self.train_loader = self.data_splits.train_loader
            self.val_loader = self.data_splits.val_loader
        
        max_len = self.data_splits.max_len
        
        if model=='tf':
            self.model = TransformerModel(input_dim, num_heads, num_layers, hidden_dim, max_len, dropout, device = device).to(device)
        elif model == 'mlp':
            self.model = MLP(input_dim, hidden_dim, output_dim).to(device)
        elif model == 'mlp_gpt':
            self.model = MLP_gpt(input_dim, hidden_dim, output_dim).to(device)
        elif model == 'lr_gpt' or model == 'lr':
            self.model = LR(input_dim).to(device)
        elif model == 'mlp_fix':
            self.model = MLP_fix(input_dim, hidden_dim, output_dim).to(device)
        else:
            print('error: choose model')
                
    def training(self, lr = 0.0001): #0.00001
        criterion = nn.BCEWithLogitsLoss()  # or any other appropriate loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)  # or any other optimizer

        num_epochs = 2000
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model = None

        epochs_without_improvement = 0
        patience = 10

        for epoch in range(num_epochs):
            epoch_train_losses = []  # to store losses for this epoch

            self.model.train()  # ensure the model is in train mode
            for sequences, labels in self.train_loader:
                sequences, labels = sequences.to(self.device), labels.float().to(self.device)
                
                seq_lengths = [len(seq[seq != 0]) for seq in sequences]  # Calculate actual lengths without padding
                max_len = sequences.size(1)
                mask = self.create_padding_mask(seq_lengths, max_len).to(self.device)

                outputs = self.model(sequences, mask)
                labels = labels.unsqueeze(1)

                loss = criterion(outputs, labels)
                epoch_train_losses.append(loss.item())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(avg_train_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss}')

            # Evaluate on validation set
            self.model.eval()  # ensure the model is in eval mode
            with torch.no_grad():
                val_losses_this_epoch = []
                for sequences, labels in self.val_loader:
                    sequences, labels = sequences.to(self.device), labels.float().to(self.device)
                    
                    seq_lengths = [len(seq[seq != 0]) for seq in sequences]  # Calculate actual lengths without padding
                    max_len = sequences.size(1)
                    mask = self.create_padding_mask(seq_lengths, max_len).to(self.device)

                    outputs = self.model(sequences, mask)
                    labels = labels.unsqueeze(1)
                    loss = criterion(outputs, labels)
                    val_losses_this_epoch.append(loss.item())

                avg_val_loss = sum(val_losses_this_epoch) / len(val_losses_this_epoch)
                val_losses.append(avg_val_loss)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss}')

            # if this model is the best so far, save it
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement +=1
                if epochs_without_improvement == patience:
                    print("Stopping training due to lack of improvement in validation loss.")
                    break

        return best_model, train_losses, val_losses
    
    def create_padding_mask(self, seq_lengths, max_len):
        batch_size = len(seq_lengths)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, seq_len in enumerate(seq_lengths):
            mask[i, seq_len:] = True
        return mask