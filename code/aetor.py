import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        
        # Set random seeds for reproducibility
        torch.manual_seed(23)
        np.random.seed(23)
        
        # Model parameters
        self.latent_dim = 196
        
        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []
        
        self.lr = 1e-4
        
        # Build encoder and decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
    def build_encoder(self):
        return nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [batch, 16, 128, 128]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # [batch, 16, 64, 64]
            
            nn.Conv2d(16, 8, kernel_size=3, padding=1),  # [batch, 8, 64, 64]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # [batch, 8, 32, 32]
            
            nn.Conv2d(8, 4, kernel_size=3, padding=1),   # [batch, 4, 32, 32]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # [batch, 4, 16, 16]
            
            nn.Flatten(),
            nn.Linear(16 * 16 * 4, self.latent_dim)      # [batch, 196]
        )
    
    def build_decoder(self):
        return nn.Sequential(
            nn.Linear(self.latent_dim, 16 * 16 * 4),     # [batch, 1024]
            nn.ReLU(inplace=True),
            
            nn.Unflatten(1, (4, 16, 16)),               # [batch, 4, 16, 16]
            
            nn.ConvTranspose2d(4, 4, kernel_size=2, stride=2),  # [batch, 4, 32, 32]
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(4, 8, kernel_size=2, stride=2),  # [batch, 8, 64, 64]
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2), # [batch, 16, 128, 128]
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 1, kernel_size=3, padding=1), # [batch, 1, 128, 128]
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Ensure input has channel dimension: [batch, 1, 128, 128]
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def loss(self, y_pred, y_true):
        # Mean Squared Error loss
        train_loss = F.mse_loss(y_pred, y_true)
        return [train_loss]
    
    def encode(self, x):
        # x - [batch_size, 1, 128, 128] or [batch_size, 128, 128]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.encoder(x)
    
    def decode(self, x):
        # x - [batch_size, latent_dim]
        decoded = self.decoder(x)
        # Remove channel dimension to match TensorFlow output: [batch, 128, 128]
        return decoded.squeeze(1)


# Example usage
if __name__ == "__main__":
    # Create model
    model = AE()
    
    # Example input
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 128)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Encode
    encoded = model.encode(x)
    print(f"Encoded shape: {encoded.shape}")
    
    # Decode
    decoded = model.decode(encoded)
    print(f"Decoded shape: {decoded.shape}")
    
    # Loss
    loss = model.loss(output, x)
    print(f"Loss: {loss[0].item()}")