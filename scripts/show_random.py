import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
from torchvision.datasets import CIFAR10

# Папка для локального хранения CIFAR-10
dataset_path = Path("./dataset")

# -------------------------
# Загружаем CIFAR-10 (download=True скачает только если нет на диске)
# -------------------------
dataset = CIFAR10(root=dataset_path, train=True, download=True)

# -------------------------
# Выбираем случайный индекс
# -------------------------
idx = random.randint(0, len(dataset) - 1)
img, _ = dataset[idx]  # PIL.Image

# RGB -> Lab
img_np = np.array(img)
lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)

# L канал (ч/б)
L_img = lab[:, :, 0] / 255.0

# ab каналы
ab_img = lab[:, :, 1:] / 128.0 - 1.0

# Восстановление RGB для визуализации
lab_restore = np.zeros_like(lab)
lab_restore[:, :, 0] = L_img * 255
lab_restore[:, :, 1:] = (ab_img + 1) * 128
rgb_img = cv2.cvtColor(lab_restore.astype(np.uint8), cv2.COLOR_LAB2RGB)

# -------------------------
# Визуализация
# -------------------------
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(L_img, cmap='gray')
plt.title('Ч/Б (L)')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(rgb_img)
plt.title('Цветное (RGB)')
plt.axis('off')

plt.show()
