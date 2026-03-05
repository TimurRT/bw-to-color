from clearml import Dataset
from torchvision.datasets import CIFAR10
import os

# скачиваем CIFAR10 локально
dataset_path = "./cifar_local"

CIFAR10(root=dataset_path, train=True, download=True)
CIFAR10(root=dataset_path, train=False, download=True)

# создаём dataset в ClearML
dataset = Dataset.create(
    dataset_name="CIFAR10",
    dataset_project="VAE_Colorization"
)

# добавляем файлы
dataset.add_files(dataset_path)

# загружаем
dataset.upload()

# фиксируем версию
dataset.finalize()

print("Dataset uploaded to ClearML")