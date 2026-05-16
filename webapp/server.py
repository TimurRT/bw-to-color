# webapp/server.py
import io
import sys
from pathlib import Path

# Добавляем корень проекта в путь (работает и с uv run)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, request, send_file, jsonify

# ============================================
# Flask app
# ============================================
app = Flask(__name__)

# ============================================
# Архитектура (копия из main.py)
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
# Загрузка моделей
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_cache = {}

def get_model(model_name):
    if model_name in models_cache:
        return models_cache[model_name]
    
    model_path = Path(__file__).parent.parent / "models" / model_name
    model = ResNetUNetGenerator(in_channels=1, out_channels=3).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    models_cache[model_name] = model
    return model

def get_available_models():
    # Путь относительно корня проекта
    models_dir = Path(__file__).parent.parent / "models"
    if not models_dir.exists():
        return []
    return [f.name for f in models_dir.glob("*.pth")]

# ============================================
# Роуты
# ============================================
@app.route("/")
def index():
    models = get_available_models()
    return render_template("index.html", models=models)

@app.route("/colorize", methods=["POST"])
def colorize():
    if "image" not in request.files:
        return "No image", 400
    
    file = request.files["image"]
    model_name = request.form.get("model", "")
    
    if not model_name:
        return "No model selected", 400
    
    # Загружаем модель
    model = get_model(model_name)
    
    # Загружаем изображение из байтов
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    original_size = img.size  # сохраняем оригинальный размер
    
    # Grayscale вход — БЕЗ изменения размера
    gray_input = img.convert('L')
    
    # Трансформация в тензор
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    gray_tensor = transform(gray_input).unsqueeze(0).to(device)
    
    # Инференс
    with torch.no_grad():
        colorized = model(gray_tensor)
    
    # Конвертируем обратно в PIL изображение
    colorized = colorized.squeeze(0).cpu()
    colorized = (colorized + 1) / 2  # [-1, 1] -> [0, 1]
    colorized = colorized.clamp(0, 1)
    colorized = transforms.ToPILImage()(colorized)
    
    # На всякий случай проверяем, что размер совпадает
    if colorized.size != original_size:
        colorized = colorized.resize(original_size, Image.BICUBIC)
    
    # Отправляем результат
    buf = io.BytesIO()
    colorized.save(buf, format="PNG")
    buf.seek(0)
    
    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Available models: {get_available_models()}")
    app.run(host="127.0.0.1", port=5000, debug=True)