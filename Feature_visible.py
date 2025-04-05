import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class OcclusionHeatmapGenerator:
    def __init__(self, model, window_size=32, stride=16, device='cuda'):
        self.model = model.eval().to(device)
        self.window_size = window_size
        self.stride = stride
        self.device = device

    def generate(self, image_tensor):
        _, _, h, w = image_tensor.shape
        original_embedding = self.model(image_tensor.to(self.device)).detach()

        heatmap = np.zeros((h, w))
        countmap = np.zeros((h, w))

        for y in range(0, h - self.window_size + 1, self.stride):
            for x in range(0, w - self.window_size + 1, self.stride):
                occluded = image_tensor.clone()
                occluded[:, :, y:y+self.window_size, x:x+self.window_size] = 0

                embedding = self.model(occluded.to(self.device)).detach()
                score_drop = F.pairwise_distance(original_embedding, embedding).item()

                heatmap[y:y+self.window_size, x:x+self.window_size] += score_drop
                countmap[y:y+self.window_size, x:x+self.window_size] += 1

        heatmap /= countmap + 1e-8
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize
        return heatmap
def overlay_heatmap(image_pil, heatmap):
    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image_pil.size)
    heatmap_colored = plt.cm.jet(np.array(heatmap_resized))[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    blended = Image.blend(image_pil.convert("RGB"), Image.fromarray(heatmap_colored), alpha=0.5)
    return blended
