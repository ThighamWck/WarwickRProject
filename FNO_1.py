# pde_operator_learning.py

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

        # Debugging print to check the columns in group_df
        #print(f"Columns in group_df: {group_df.columns}")

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
        # Initialize weights with correct shape
       # self.weights = nn.Parameter(torch.randn(in_channels, out_channels, modes1, modes2)  # [64, 64, 16, 16] or desired shape
        # Initialize weights with shape [1, 64, modes1, modes2] to allow broadcasting across batch size
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

        # Debugging print statement for tensor shape before Fourier layers
       # print(f"Model input shape after fc0: {x.shape}")

        # Apply Fourier layers
        for layer in self.fourier_layers:
            x = F.gelu(layer(x))
####################### Change
        # residual = x
        # for layer in self.fourier_layers:
        #     x = F.gelu(layer(x)) + residual

        # Final linear layers
        x = x.permute(0, 2, 3, 1)  # [B, 16, 16, width]
        # x = self.fc1(x).gelu() #changed to gelu from Relu
        x = F.gelu(self.fc1(x))  # Correct: applying GELU via the functional API

        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)  # [B, out_channels, 16, 16]


# ===============================
# LOAD CSV AND CREATE DATASET
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

####new
    # Learning rate scheduler: halve the learning rate every 100 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100) #,  gamma=0.5)

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
        
    #     #####new
    # # Only step the learning rate scheduler every 100 epochs
    #     if (epoch + 1) % 100 == 0:
    #         lr_scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.6f}")

# ===============================
# MAIN FUNCTION
# ===============================
def main():
    # Path to your CSV file
    csv_path = "rho_16_square_train_1000.csv"  # Updated path to your data

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

# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import copy

# # Device Setup
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")

# # Fourier Layer with Consistent Dropout
# class FourierLayer2D(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2, dropout_rate=0.2):
#         super(FourierLayer2D, self).__init__()
#         self.modes1, self.modes2 = modes1, modes2
        
#         # Regularized weight initialization
#         self.weights = nn.Parameter(
#             torch.randn(1, out_channels, modes1, modes2) * 0.02
#         )
        
#         # Consistent dropout
#         self.dropout = nn.Dropout2d(p=dropout_rate)

#     def forward(self, x):
#         # Fourier transform processing
#         x_ft = torch.fft.fftn(x, dim=(-2, -1))
        
#         # Apply weights 
#         out_ft = torch.zeros_like(x_ft)
#         out_ft[:, :, :self.modes1, :self.modes2] = x_ft[:, :, :self.modes1, :self.modes2] * self.weights
        
#         # Inverse Fourier transform
#         out = torch.fft.ifftn(out_ft, dim=(-2, -1)).real
        
#         # Apply consistent dropout
#         out = self.dropout(out)
        
#         return out

# # Enhanced FNO Model with Consistent Dropout
# class FNO2D(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2, width, dropout_rate=0.2):
#         super(FNO2D, self).__init__()
        
#         # Input projection with consistent dropout
#         self.fc0 = nn.Sequential(
#             nn.Linear(in_channels, width),
#             nn.GELU(),
#             nn.Dropout(dropout_rate)
#         )
        
#         # Fourier layers with consistent dropout
#         self.fourier_layers = nn.ModuleList([
#             FourierLayer2D(width, width, modes1, modes2, dropout_rate) 
#             for _ in range(4)
#         ])
        
#         # Regularized final layers with consistent dropout
#         self.fc1 = nn.Sequential(
#             nn.Linear(width, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )
#         self.fc2 = nn.Linear(128, out_channels)
        
#         # Weight initialization
#         self.init_weights()

#     def init_weights(self):
#         # Xavier initialization for linear layers
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x):
#         # Initial linear transform with dropout
#         x = self.fc0(x.permute(0, 2, 3, 1))
#         x = x.permute(0, 3, 1, 2)

#         # Apply Fourier layers with consistent dropout
#         for layer in self.fourier_layers:
#             x = F.gelu(layer(x))

#         # Final linear layers with dropout
#         x = x.permute(0, 2, 3, 1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x.permute(0, 3, 1, 2)
# # Training Function with Regularization
# def train_model(model, dataloader, num_epochs=100, lr=0.001, patience=10):
#     # Optimizer with weight decay (L2 regularization)
#     optimizer = optim.AdamW(
#         model.parameters(), 
#         lr=lr, 
#         weight_decay=1e-5  # Added weight decay
#     )
    
#     # Learning rate scheduler 
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer,
#         mode='min',
#         factor=0.5,
#         patience=5,
#         min_lr=1e-6
#     )
    
#     # Loss function
#     criterion = nn.MSELoss(reduction='mean')
    
#     # Early stopping and model checkpointing
#     best_loss = float('inf')
#     patience_counter = 0
#     best_model = None
    
#     for epoch in range(num_epochs):
#         # Training phase
#         model.train()
#         total_loss = 0.0
        
#         for batch_idx, (x, y) in enumerate(dataloader):
#             x, y = x.to(device), y.to(device).unsqueeze(1)
            
#             # Zero gradients
#             optimizer.zero_grad()
            
#             # Forward pass
#             outputs = model(x)
#             loss = criterion(outputs, y)
            
#             # Backward pass with gradient clipping
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
            
#             total_loss += loss.item()
        
#         # Average loss
#         avg_loss = total_loss / len(dataloader)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        
#         # Learning rate scheduling
#         scheduler.step(avg_loss)
        
#         # Early stopping logic
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             patience_counter = 0
#             best_model = copy.deepcopy(model.state_dict())
#         else:
#             patience_counter += 1
        
#         # Stop if patience exceeded
#         if patience_counter >= patience:
#             print(f"Early stopping triggered. Best loss: {best_loss:.6f}")
#             break
    
#     # Restore best model
#     if best_model:
#         model.load_state_dict(best_model)
    
#     return model

# # Data Loading Function
# def load_data(csv_path, grid_size=63, batch_size=64):
#     # Load CSV
#     data = pd.read_csv(csv_path)

#     # Group by solution_id
#     grouped_data = data.groupby('solution_id')

#     # Create dataset and dataloader
#     dataset = PDEOperatorDataset(grouped_data, grid_size)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     return dataloader

# # Dataset Class
# class PDEOperatorDataset(Dataset):
#     def __init__(self, grouped_data, grid_size=63):
#         self.grouped_data = grouped_data
#         self.grid_size = grid_size

#     def __len__(self):
#         return len(self.grouped_data)

#     def __getitem__(self, idx):
#         # Get the group (PDE solution) based on index
#         group_df = self.grouped_data.get_group(idx)

#         # Extract input features (x, y, b1, b2) and target (rho)
#         x = torch.tensor(group_df[['x', 'y', 'b1', 'b2']].values, dtype=torch.float32)

#         # Extract 'rho' as the target variable
#         if 'rho' in group_df.columns:
#             target = torch.tensor(group_df['rho'].values, dtype=torch.float32)
#         else:
#             raise ValueError("'rho' column not found in group_df")

#         # Reshape to grid format
#         x = x.view(self.grid_size, self.grid_size, -1).permute(2, 0, 1)  # [4, 16, 16]
#         target = target.view(self.grid_size, self.grid_size)

#         return x, target

# # Main function
# def main():
#     csv_path = "rho_63_square_train_10000.csv"
    
#     # Load full dataset
#     dataloader = load_data(csv_path)
    
#     # Initialize model with regularization
#     model = FNO2D(
#         in_channels=4, 
#         out_channels=1, 
#         modes1=20, 
#         modes2=20, 
#         width=128, 
#         dropout_rate=0.2
#     ).to(device)
    
#     # Train model
#     trained_model = train_model(model, dataloader)
    
#     # Save the model
#     torch.save(trained_model.state_dict(), 'rho_63_square_10000_regularized.pth')
#     print('Regularized model saved successfully!')

# if __name__ == "__main__":
#     main()
