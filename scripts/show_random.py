# scripts/test_local.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Импорт архитектуры
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# ============================================
# ПУТЬ К МОДЕЛИ (укажите ваш путь)
# ============================================
MODEL_PATH = "models/generator_final.pth"  # ← ваш путь к модели

# Папка с тестовыми изображениями
TEST_FOLDER = Path("test_images")

# ============================================
# Архитектура генератора
# ============================================
class ResNetUNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        old_conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight[:, :1] = old_conv1.weight.mean(dim=1, keepdim=True)
        
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(256 + 256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128 + 128, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(64 + 64, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final = nn.Sequential(
            nn.Conv2d(32 + 64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, padding=1),
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
        input_size = x.shape[2:]
        x0 = self.initial(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        d1 = self.up1(x4)
        d1 = F.interpolate(d1, size=x3.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d1, x3], dim=1))
        d2 = self.up2(d1)
        d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))
        d3 = self.up3(d2)
        d3 = F.interpolate(d3, size=x1.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d3, x1], dim=1))
        d4 = self.up4(d3)
        d4 = F.interpolate(d4, size=x0.shape[2:], mode='bilinear', align_corners=True)
        out = self.final(torch.cat([d4, x0], dim=1))
        return F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

# ============================================
# Функция раскрашивания
# ============================================
def colorize_image(model, image_path, device):
    img = Image.open(image_path).convert('RGB')
    
    # Grayscale вход
    gray = img.convert('L').resize((256, 256))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    gray_tensor = transform(gray).unsqueeze(0).to(device)
    
    # Раскрашиваем
    with torch.no_grad():
        colorized = model(gray_tensor)
    
    # Конвертируем обратно в изображение
    colorized = colorized.squeeze(0).cpu()
    colorized = (colorized + 1) / 2  # [-1, 1] -> [0, 1]
    colorized = colorized.clamp(0, 1)
    colorized = transforms.ToPILImage()(colorized)
    
    return gray, colorized, img.resize((256, 256))

# ============================================
# Главная функция
# ============================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Проверяем модель
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        print(f"ERROR: Model not found at {model_file.absolute()}")
        print("Please update MODEL_PATH in the script")
        return
    
    # Загружаем модель
    print(f"Loading model: {model_file}")
    model = ResNetUNetGenerator(in_channels=1, out_channels=3).to(device)
    state_dict = torch.load(str(model_file), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded!")
    
    # Ищем тестовые изображения
    TEST_FOLDER.mkdir(exist_ok=True)
    
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        test_images.extend(TEST_FOLDER.glob(ext))
    
    if not test_images:
        print(f"\nNo test images found in {TEST_FOLDER.absolute()}")
        print("Please put some color images in the 'test_images' folder")
        return
    
    test_images = test_images[:3]
    print(f"\nFound {len(test_images)} test images:")
    for img in test_images:
        print(f"  - {img.name}")
    
    # Обрабатываем изображения
    print("\nColorizing...")
    fig, axes = plt.subplots(len(test_images), 3, figsize=(12, 4 * len(test_images)))
    if len(test_images) == 1:
        axes = axes.reshape(1, -1)
    
    for i, img_path in enumerate(test_images):
        gray, colorized, original = colorize_image(model, str(img_path), device)
        
        axes[i, 0].imshow(gray, cmap='gray')
        axes[i, 0].set_title(f'Input: {img_path.name}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(colorized)
        axes[i, 1].set_title('Colorized')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(original)
        axes[i, 2].set_title('Original')
        axes[i, 2].axis('off')
    
    # Сохраняем результат
    plt.tight_layout()
    output_path = Path("test_result.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResult saved to: {output_path.absolute()}")
    plt.show()

if __name__ == "__main__":
    main()