import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from clearml import Task, Dataset

from data.cifar_dataset import CIFARColorization
from models.vae import VAE
from utils.visualization import visualize_colorization


def main():

    # -------------------------
    # ClearML task
    # -------------------------
    task = Task.init(
        project_name="VAE_Colorization",
        task_name="CIFAR10_Experiment"
    )

    print("ClearML task initialized.")

    # -------------------------
    # Load dataset from ClearML
    # -------------------------
    dataset = Dataset.get(
        dataset_name="CIFAR10",
        dataset_project="VAE_Colorization"
    )

    dataset_path = dataset.get_local_copy()

    print("Dataset downloaded to:", dataset_path)

    # -------------------------
    # Create results folder
    # -------------------------
    os.makedirs("results", exist_ok=True)

    # -------------------------
    # PyTorch datasets
    # -------------------------
    train_dataset = CIFARColorization(
        root=dataset_path,
        train=True
    )

    val_dataset = CIFARColorization(
        root=dataset_path,
        train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    task.get_logger().report_text(
        f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}"
    )

    # -------------------------
    # Device
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -------------------------
    # Model
    # -------------------------
    latent_dim = 32

    model = VAE(latent_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss(reduction="mean")

    epochs = 5

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(epochs):

        model.train()
        train_loss = 0

        for L, ab in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):

            L = L.to(device)
            ab = ab.to(device)

            optimizer.zero_grad()

            output_ab, mu, logvar = model(L)

            mse = mse_loss(output_ab, ab)
            kl = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )

            loss = mse + kl

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * L.size(0)

        train_loss /= len(train_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")

        task.get_logger().report_scalar(
            "Loss/train",
            "VAE Loss",
            iteration=epoch + 1,
            value=train_loss
        )

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():

            for L, ab in val_loader:

                L = L.to(device)
                ab = ab.to(device)

                output_ab, mu, logvar = model(L)

                mse = mse_loss(output_ab, ab)
                kl = -0.5 * torch.mean(
                    1 + logvar - mu.pow(2) - logvar.exp()
                )

                loss = mse + kl

                val_loss += loss.item() * L.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f}")

        task.get_logger().report_scalar(
            "Loss/val",
            "VAE Loss",
            iteration=epoch + 1,
            value=val_loss
        )

        # -------------------------
        # Visualization
        # -------------------------
        visualize_colorization(
            model,
            val_loader,
            device,
            epoch + 1
        )

    # -------------------------
    # Save model
    # -------------------------
    torch.save(model.state_dict(), "results/vae_model.pth")

    print("Model saved to results/vae_model.pth")


if __name__ == "__main__":
    main()