
"""
#################### 
# Fourier Neural Operator trainer for square domains - for Fourth Year Project
# Author - Thomas Higham
# Date - 31/03/25
# University of Warwick
#####################
Initialisations is in main function at bottom
# """

# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from functools import partial
from torch.fft import fft2, ifft2


# ===============================
# DEVICE SETUP (MPS or CPU)
# ===============================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# ===============================
# LOAD AND PREPARE DATA
"""Specify gridsize here """
# ===============================
class PDEOperatorDataset(Dataset):
    def __init__(self, grouped_data, grid_size=16):
        self.grouped_data = grouped_data
        self.grid_size = grid_size

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        # Get the group (PDE solution) based on index
        group_df = self.grouped_data.get_group(idx)


        # Extract input features (x, y, b1, b2) and target (rho)
        x = torch.tensor(group_df[['x', 'y', 'b1', 'b2']].values, dtype=torch.float32)

        # Extract 'rho' as the target variable
        if 'rho' in group_df.columns:
            target = torch.tensor(group_df['rho'].values, dtype=torch.float32)
        else:
            raise ValueError("'rho' column not found in group_df")

        # Reshape to grid format (16x16)
        x = x.view(self.grid_size, self.grid_size, -1).permute(2, 0, 1)  # [4, 16, 16]
        target = target.view(self.grid_size, self.grid_size)

        return x, target

# ===============================
# DEFINE FOURIER LAYER
# ===============================
class FourierLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(FourierLayer2D, self).__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(torch.randn(1, out_channels, modes1, modes2)  # Shape [1, 64, 12, 12] or similar
        )

    def forward(self, x):
            # Fourier transform processing (FFT)
            x_ft = torch.fft.fftn(x, dim=(-2, -1))
            
            # Apply weights (make sure the shapes match)
            out_ft = torch.zeros_like(x_ft)
            out_ft[:, :, :self.modes1, :self.modes2] = x_ft[:, :, :self.modes1, :self.modes2] * self.weights
            
            # Inverse Fourier transform
            out = torch.fft.ifftn(out_ft, dim=(-2, -1)).real
            
            return out

# ===============================
# DEFINE FNO MODEL
# ===============================
class FNO2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, width):
        super(FNO2D, self).__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.fourier_layers = nn.ModuleList([
            FourierLayer2D(width, width, modes1, modes2) for _ in range(4)
        ])
        self.fc1 = nn.Linear(width, 128) #was 128
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # Initial linear transform
        x = self.fc0(x.permute(0, 2, 3, 1))  # [B, 16, 16, in_channels] -> [B, 16, 16, width]
        x = x.permute(0, 3, 1, 2)  # [B, width, 16, 16]


        # Apply Fourier layers
        for layer in self.fourier_layers:
            x = F.gelu(layer(x))
        # Final linear layers
        x = x.permute(0, 2, 3, 1)  # [B, 16, 16, width]
        x = F.gelu(self.fc1(x))  #Applying GELU

        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)  # [B, out_channels, 16, 16]


# ===============================
# LOAD CSV AND CREATE DATASET
"""Specify gridsize here """
# ===============================
def load_data(csv_path, grid_size=16):
    # Load CSV
    data = pd.read_csv(csv_path)

    # Group by solution_id
    grouped_data = data.groupby('solution_id')

    
    # Create dataset and dataloader
    dataset = PDEOperatorDataset(grouped_data, grid_size)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    return dataloader


# ===============================
# TRAINING LOOP
# ===============================
def train_model(model, dataloader, num_epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()


    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device).unsqueeze(1)

            # Forward pass
            outputs = model(x)
            loss = criterion(outputs, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.6f}")

# ===============================
# MAIN FUNCTION
# ===============================
def main():
    # Path to CSV file
    csv_path = "rho_16_square_train_1000.csv"  

    # Load data
    dataloader = load_data(csv_path)

    # Initialize model
    model = FNO2D(in_channels=4, out_channels=1, modes1=12, modes2=12, width=128).to(device)

    # Train the model
    train_model(model, dataloader)

    # Save the model after training
    torch.save(model.state_dict(), 'rho_16_square_1000_Test_1.pth')
    print('Model saved successfully!')

# Run main if script is executed
if __name__ == "__main__":
    main()

