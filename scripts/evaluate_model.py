# scripts/evaluate_model.py
from clearml import Task, InputModel, Logger
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Инициализация отдельной задачи для тестирования
task = Task.init(
    project_name="Colorization_GAN",
    task_name="Evaluate model",
    task_type=Task.TaskTypes.testing
)

# 2. Загрузка обученной модели из ClearML по ID (замените на ID вашей модели)
input_model = InputModel(model_id="your_model_id_here") 
local_model_path = input_model.get_local_copy()

# 3. Ваш код для загрузки модели и тестовых данных...
# generator = YourGeneratorClass()
# generator.load_state_dict(torch.load(local_model_path))
# generator.eval()

# 4. Создание и логирование примеров колоризации
logger = task.get_logger()
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for i, test_image in enumerate(test_images[:3]):
    # Ваш код колоризации: output_image = generator(test_image)
    
    # Отрисовка черно-белого оригинала и цветного результата
    axes[0, i].imshow(test_image.squeeze(), cmap='gray')
    axes[0, i].set_title('Input (Grayscale)')
    axes[1, i].imshow(output_image)
    axes[1, i].set_title('Colorized Output')

# 5. Отправка результата в ClearML (вкладка PLOTS и DEBUG SAMPLES)
logger.report_matplotlib_figure(
    title="Colorization Examples",
    series="Results",
    figure=fig,
    report_image=True # Сохранит как изображение во вкладке DEBUG SAMPLES
)

print("Оценка завершена. Результаты доступны в веб-интерфейсе ClearML.")
task.close()