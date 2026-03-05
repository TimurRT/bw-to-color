import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2


def lab_to_rgb(L, ab):
    """
    Конвертирует LAB обратно в RGB
    """
    L = L * 255.0
    ab = (ab + 1) * 128.0

    lab = np.zeros((32, 32, 3))
    lab[:, :, 0] = L
    lab[:, :, 1:] = ab

    lab = lab.astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return rgb


def visualize_colorization(model, dataloader, device, epoch, save_path="results"):
    model.eval()

    with torch.no_grad():
        L, ab = next(iter(dataloader))

        L = L.to(device)

        pred_ab, _, _ = model(L)

        L = L.cpu().numpy()
        ab = ab.numpy()
        pred_ab = pred_ab.cpu().numpy()

        fig, axes = plt.subplots(3, 5, figsize=(10, 6))

        for i in range(5):

            L_img = L[i][0]
            ab_gt = ab[i].transpose(1, 2, 0)
            ab_pred = pred_ab[i].transpose(1, 2, 0)

            gt_rgb = lab_to_rgb(L_img, ab_gt)
            pred_rgb = lab_to_rgb(L_img, ab_pred)

            axes[0, i].imshow(L_img, cmap="gray")
            axes[0, i].set_title("Input")

            axes[1, i].imshow(pred_rgb)
            axes[1, i].set_title("Pred")

            axes[2, i].imshow(gt_rgb)
            axes[2, i].set_title("GT")

            for j in range(3):
                axes[j, i].axis("off")

        plt.tight_layout()
        plt.savefig(f"{save_path}/epoch_{epoch}.png")
        plt.close()