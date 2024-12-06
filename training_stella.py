from split_data import DataSplit

from model import*
import torch
import torch.nn as nn
import copy

from split_data_stella import DataSplit
from model import *
import torch
import torch.nn as nn
import copy

from split_data_stella import DataSplit
from model import *
import torch
import torch.nn as nn
import copy


class Training:
    def __init__(self, data=None, model=None, input_dim=1, hidden_dim=128, output_dim=1, num_heads=4, num_layers=2, dropout=0.1, device=torch.device('cpu')):
        self.device = device

        # Convert dictionary to a list of dictionaries with 'emb' and 'label' keys
        if isinstance(data, dict):
            data = [{'emb': value['emb'], 'label': value['label']} for key, value in data.items()]

        # Validate data structure
        if not isinstance(data, list) or not all('emb' in item and 'label' in item for item in data):
            raise ValueError("Data must be a list of dictionaries with 'emb' and 'label' keys.")
        if len(data[0]['emb']) != input_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {input_dim}, got {len(data[0]['emb'])}")

        # Initialize DataSplit
        self.data_splits = DataSplit(data)
        self.train_loader = self.data_splits.train_loader
        self.val_loader = self.data_splits.val_loader

        # Initialize the model based on the model name
        if model == 'tf':
            self.model = TransformerModel(input_dim, num_heads, num_layers, hidden_dim, max_len=None, dropout=dropout, device=device).to(device)
        elif model == 'mlp':
            self.model = MLP(input_dim, hidden_dim, output_dim).to(device)
        elif model == 'mlp_gpt':
            self.model = MLP_gpt(input_dim, hidden_dim, output_dim).to(device)
        elif model in ['lr_gpt', 'lr']:
            self.model = LR(input_dim).to(device)
        elif model == 'mlp_fix':
            self.model = MLP_fix(input_dim, hidden_dim, output_dim).to(device)
        else:
            raise ValueError("Error: Choose a valid model")

    def training(self, lr=0.0001, num_epochs=2000, patience=10):
        """
        Train the model with the given data and parameters.

        :param lr: Learning rate for the optimizer.
        :param num_epochs: Maximum number of epochs for training.
        :param patience: Number of epochs to wait for improvement before stopping early.
        :return: Best model's state_dict, training losses, and validation losses.
        """
        criterion = nn.BCEWithLogitsLoss()  # Loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)  # Optimizer

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model = None

        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            epoch_train_losses = []
            for embeddings, labels in self.train_loader:
                embeddings, labels = embeddings.to(self.device), labels.float().to(self.device)
                labels = labels.unsqueeze(1)  # Ensure labels match output dimensions
                mask = (embeddings != 0).any(dim=1).unsqueeze(-1)
                mask = mask.float()
                mask = mask.bool().unsqueeze(-1) 
                outputs = self.model(embeddings,mask)
                loss = criterion(outputs, labels)
                epoch_train_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(avg_train_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_losses_this_epoch = []
                for embeddings, labels in self.val_loader:
                    embeddings, labels = embeddings.to(self.device), labels.float().to(self.device)
                    labels = labels.unsqueeze(1)

                    outputs = self.model(embeddings,mask)
                    loss = criterion(outputs, labels)
                    val_losses_this_epoch.append(loss.item())

                avg_val_loss = sum(val_losses_this_epoch) / len(val_losses_this_epoch)
                val_losses.append(avg_val_loss)
                print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Stopping training due to lack of improvement in validation loss.")
                    break

        return best_model, train_losses, val_losses


