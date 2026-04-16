# main.py
import argparse
import random
from pathlib import Path

import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from clearml import Task, Dataset as ClearMLDataset, Logger, OutputModel

# ============================================
# 1. Инициализация ClearML Task
# ============================================
task = Task.init(
    project_name="Colorization_GAN",
    task_name="GAN_training",
    task_type=Task.TaskTypes.training,
    auto_connect_frameworks={'pytorch': True}
)

# ============================================
# 2. Гиперпараметры (можно менять в UI ClearML)
# ============================================
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
parser.add_argument("--image_size", type=int, default=256, help="Image size")
parser.add_argument("--lambda_l1", type=float, default=100.0, help="L1 loss weight")
parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
args = parser.parse_args()
task.connect(args)

# Фиксируем random seed для воспроизводимости
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = task.get_logger()

# ============================================
# 3. Загрузка датасета из ClearML
# ============================================
print("=== Загрузка датасета из ClearML ===")
dataset_path = ClearMLDataset.get(
    dataset_name="coco_2017_train",  # или cifar10_colorization
    dataset_project="Colorization_GAN"
).get_local_copy()
print(f"Датасет загружен в: {dataset_path}")

# ============================================
# 4. Аугментация и Dataset класс
# ============================================
class PairedAugmentation:
    """Синхронная аугментация для пары (grayscale, color)"""
    def __init__(self, img_size=256):
        self.img_size = img_size
        
    def __call__(self, input_img, target_img):
        # Resize
        resize = transforms.Resize((self.img_size, self.img_size))
        input_img = resize(input_img)
        target_img = resize(target_img)

        # Random Horizontal Flip
        if random.random() > 0.5:
            input_img = TF.hflip(input_img)
            target_img = TF.hflip(target_img)

        # Random Rotation (-15 to 15 degrees)
        angle = random.uniform(-15, 15)
        input_img = TF.rotate(input_img, angle)      # ← angle добавлен
        target_img = TF.rotate(target_img, angle)    # ← angle добавлен

        # Color Jitter ONLY on target (color image)
        color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        )
        target_img = color_jitter(target_img)

        return input_img, target_img

class ResNetUNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        
        # 1. Загружаем предобученный ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 2. Адаптируем первый слой под 1 канал (grayscale)
        old_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight[:, :1] = old_conv1.weight.mean(dim=1, keepdim=True)
        
        # 3. Разделяем ResNet на блоки для skip-connections
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )  # Выход: 64 канала, размер H/4, W/4
        
        self.layer1 = resnet.layer1  # Выход: 64 канала
        self.layer2 = resnet.layer2  # Выход: 128 каналов
        self.layer3 = resnet.layer3  # Выход: 256 каналов
        self.layer4 = resnet.layer4  # Выход: 512 каналов
        
        # 4. Замораживаем энкодер (опционально)
        for param in self.initial.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False
            
        # 5. Декодер с правильными размерами каналов
        # После layer4: 512 каналов, размер H/32, W/32
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(256 + 256, 256)  # up1(256) + layer3(256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128 + 128, 128)  # up2(128) + layer2(128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(64 + 64, 64)     # up3(64) + layer1(64)
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # Для финального слоя нужен skip от initial блока (64 канала)
        self.final = nn.Sequential(
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Сохраняем оригинальный размер для финального выравнивания
        input_size = x.shape[2:]
        
        # Энкодер
        x0 = self.initial(x)      # 64 канала,  H/4
        x1 = self.layer1(x0)      # 64 канала,  H/4
        x2 = self.layer2(x1)      # 128 каналов, H/8
        x3 = self.layer3(x2)      # 256 каналов, H/16
        x4 = self.layer4(x3)      # 512 каналов, H/32
        
        # Декодер с skip-connections
        d1 = self.up1(x4)                            # 256 каналов, H/16
        d1 = F.interpolate(d1, size=x3.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, x3], dim=1))   # 256 каналов
        
        d2 = self.up2(d1)                            # 128 каналов, H/8
        d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))   # 128 каналов
        
        d3 = self.up3(d2)                            # 64 канала, H/4
        d3 = F.interpolate(d3, size=x1.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, x1], dim=1))   # 64 канала
        
        d4 = self.up4(d3)                            # 32 канала, H/2
        d4 = F.interpolate(d4, size=x0.shape[2:], mode='bilinear', align_corners=True)
        out = self.final(torch.cat([d4, x0], dim=1))
        
        # Восстанавливаем оригинальный размер
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        return out
    
class ColorizationDataset(Dataset):
    def __init__(self, root_dir, image_size=256, is_train=True):
        self.image_paths = list(Path(root_dir).rglob("*.jpg"))
        # ... (код для *.png если CIFAR) ...
        
        # --- ВОТ ЭТА СТРОКА ОГРАНИЧИВАЕТ ДАТАСЕТ ---
        # Перемешиваем и берем первые 8000 для обучения
        if is_train:
            random.shuffle(self.image_paths)
            self.image_paths = self.image_paths[:8000] 
        # -----------------------------------------
        
        self.image_size = image_size
        self.is_train = is_train
        self.augmentation = PairedAugmentation(image_size) if is_train else None
        
        # Базовые трансформации
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # для grayscale
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Загружаем цветное изображение
        color_img = Image.open(img_path).convert("RGB")
        
        # Конвертируем в grayscale для входа
        gray_img = color_img.convert("L")
        
        if self.is_train and self.augmentation:
            gray_img, color_img = self.augmentation(gray_img, color_img)
        
        # Преобразуем в тензоры
        gray_tensor = self.to_tensor(gray_img)
        gray_tensor = self.normalize(gray_tensor)  # [-1, 1]
        
        color_tensor = self.to_tensor(color_img)
        color_tensor = color_tensor * 2 - 1  # [-1, 1]
        
        return gray_tensor, color_tensor

# ============================================
# 5. Архитектура GAN (упрощенный Pix2Pix)
# ============================================
class UNetGenerator(nn.Module):
    """U-Net для колоризации с правильным согласованием размеров"""
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        
        # Encoder (без пулинга внутри блоков — используем отдельные слои)
        self.enc1 = self._conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = self._conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = self._conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 512)
        
        # Decoder с правильными skip-connections
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 256)  # 512 (up4) + 512 (skip enc4)
        
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 128)   # 256 (up3) + 256 (skip enc3)
        
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 64)    # 128 (up2) + 128 (skip enc2)
        
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),  # 64 (up1) + 64 (skip enc1) = 128
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        # Обрезаем e4 если размеры не совпадают
        if d4.shape != e4.shape:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return d1

class PatchGAN(nn.Module):
    """Дискриминатор PatchGAN"""
    def __init__(self, in_channels=4):  # grayscale (1) + color (3) = 4
        super().__init__()
        
        def conv_block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            conv_block(in_channels, 64, normalize=False),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1)
        )
    
    def forward(self, gray, color):
        x = torch.cat([gray, color], dim=1)
        return self.model(x)

# ============================================
# 6. Инициализация моделей и оптимизаторов
# ============================================
generator = ResNetUNetGenerator(in_channels=1, out_channels=3).to(device)
discriminator = PatchGAN(in_channels=4).to(device)

# Размораживаем слои энкодера для тонкой настройки
# В ResNetUNetGenerator энкодер разделён на отдельные блоки
for param in generator.initial.parameters():
    param.requires_grad = True
for param in generator.layer1.parameters():
    param.requires_grad = True
for param in generator.layer2.parameters():
    param.requires_grad = True
for param in generator.layer3.parameters():
    param.requires_grad = True
for param in generator.layer4.parameters():
    param.requires_grad = True

# Для слоёв энкодера устанавливаем скорость обучения в 10 раз меньше
opt_G = optim.Adam([
    {'params': generator.initial.parameters(), 'lr': args.lr * 0.1},
    {'params': generator.layer1.parameters(), 'lr': args.lr * 0.1},
    {'params': generator.layer2.parameters(), 'lr': args.lr * 0.1},
    {'params': generator.layer3.parameters(), 'lr': args.lr * 0.1},
    {'params': generator.layer4.parameters(), 'lr': args.lr * 0.1},
    {'params': generator.up1.parameters()},
    {'params': generator.dec1.parameters()},
    {'params': generator.up2.parameters()},
    {'params': generator.dec2.parameters()},
    {'params': generator.up3.parameters()},
    {'params': generator.dec3.parameters()},
    {'params': generator.up4.parameters()},
    {'params': generator.final.parameters()},
], lr=args.lr, betas=(0.5, 0.999))

opt_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Losses
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

# ============================================
# 7. DataLoaders
# ============================================
train_dataset = ColorizationDataset(dataset_path, args.image_size, is_train=True)
train_loader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True
)

print(f"Найдено {len(train_dataset)} изображений для обучения")

# ============================================
# 8. Цикл обучения
# ============================================
print("=== Начало обучения ===")
patch_size = None

for epoch in range(args.epochs):
    epoch_loss_G = 0.0
    epoch_loss_D = 0.0
    
    for i, (gray_imgs, color_imgs) in enumerate(train_loader):
        gray_imgs = gray_imgs.to(device)
        color_imgs = color_imgs.to(device)
        
        batch_size = gray_imgs.size(0)
        
        # Динамическое определение размера патча
        if patch_size is None:
            with torch.no_grad():
                dummy_output = discriminator(gray_imgs[:1], color_imgs[:1])
                patch_size = dummy_output.shape[-1]
                print(f"Patch size detected: {patch_size}x{patch_size}")
        
        real_labels = torch.ones(batch_size, 1, patch_size, patch_size).to(device) * 0.9
        fake_labels = torch.zeros(batch_size, 1, patch_size, patch_size).to(device)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        opt_G.zero_grad()
        
        fake_color = generator(gray_imgs)
        fake_pred = discriminator(gray_imgs, fake_color)
        
        loss_G_GAN = criterion_GAN(fake_pred, real_labels)
        loss_G_L1 = criterion_L1(fake_color, color_imgs) * args.lambda_l1
        loss_G = loss_G_GAN + loss_G_L1
        
        loss_G.backward(retain_graph=True)  # ← retain_graph=True!
        opt_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        opt_D.zero_grad()
        
        # Real loss
        real_pred = discriminator(gray_imgs, color_imgs)
        loss_D_real = criterion_GAN(real_pred, real_labels)
        
        # Fake loss (используем detach() чтобы не трогать генератор)
        fake_pred = discriminator(gray_imgs, fake_color.detach())
        loss_D_fake = criterion_GAN(fake_pred, fake_labels)
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        opt_D.step()
        
        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()
        
        # Логирование каждые 100 батчей
        if i % 100 == 0:
            logger.report_scalar(
                title="Loss", 
                series="Generator", 
                value=loss_G.item(), 
                iteration=epoch * len(train_loader) + i
            )
            logger.report_scalar(
                title="Loss", 
                series="Discriminator", 
                value=loss_D.item(), 
                iteration=epoch * len(train_loader) + i
            )
    
    # Средние потери за эпоху
    avg_loss_G = epoch_loss_G / len(train_loader)
    avg_loss_D = epoch_loss_D / len(train_loader)
    
    print(f"Epoch [{epoch+1}/{args.epochs}] - Loss G: {avg_loss_G:.4f}, Loss D: {avg_loss_D:.4f}")
    
    # Сохраняем чекпоинт каждые 5 эпох
    if (epoch + 1) % 5 == 0:
        checkpoint_path = f"models/checkpoint_epoch_{epoch+1}.pth"
        Path("models").mkdir(exist_ok=True)
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'opt_G_state_dict': opt_G.state_dict(),
            'opt_D_state_dict': opt_D.state_dict(),
        }, checkpoint_path)
        
        # Загружаем чекпоинт как артефакт в ClearML
        task.upload_artifact(
            name=f"checkpoint_epoch_{epoch+1}",
            artifact_object=checkpoint_path
        )

# ============================================
# 9. Сохранение финальной модели
# ============================================
print("=== Сохранение финальной модели в ClearML ===")

final_model_path = "models/generator_final.pth"
torch.save(generator.state_dict(), final_model_path)

output_model = OutputModel(task=task, framework="PyTorch")
output_model.update_weights(weights_filename=final_model_path)

task.close()
print("Обучение завершено! Модель сохранена в ClearML.")