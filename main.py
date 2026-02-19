import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data.cifar_dataset import CIFARColorization
from models.vae import VAE
from clearml import Task

# инициализируем задачу в clearml
task = Task.init(project_name="VAE_Colorization", task_name="CIFAR10_Experiment")
print("ClearML task initialized.")

train_dataset = CIFARColorization(train=True)
val_dataset = CIFARColorization(train=False)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

task.get_logger().report_text(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

latent_dim = 32
model = VAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse_loss = nn.MSELoss(reduction='mean')

# тренировка
epochs = 5
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for L, ab in train_loader:
        L, ab = L.to(device), ab.to(device)
        optimizer.zero_grad()
        output_ab, mu, logvar = model(L)
        # VAE loss = MSE + KL
        mse = mse_loss(output_ab, ab)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = mse + kl
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * L.size(0)
    
    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
    task.get_logger().report_scalar("Loss/train", "VAE Loss", iteration=epoch+1, value=train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for L, ab in val_loader:
            L, ab = L.to(device), ab.to(device)
            output_ab, mu, logvar = model(L)
            mse = mse_loss(output_ab, ab)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + kl
            val_loss += loss.item() * L.size(0)
    val_loss /= len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f}")
    task.get_logger().report_scalar("Loss/val", "VAE Loss", iteration=epoch+1, value=val_loss)

