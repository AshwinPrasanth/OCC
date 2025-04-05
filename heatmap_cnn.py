import torch.nn as nn

class HeatmapFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=256):
        super(HeatmapFeatureExtractor, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Heatmaps are single channel
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28

            nn.AdaptiveAvgPool2d(1),  # Global average
        )
        self.embedding = nn.Linear(64, embedding_dim)

    def forward(self, heatmap):
        x = self.network(heatmap)
        x = x.view(x.size(0), -1)
        return self.embedding(x)

def preprocess_heatmap(heatmap_np):
    # Normalize and convert to tensor
    heatmap_tensor = torch.tensor(heatmap_np, dtype=torch.float32)
    heatmap_tensor = heatmap_tensor.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    return heatmap_tensor
def export_heatmap_model(model, path="heatmap_feature_extractor.pt"):
    dummy_heatmap = torch.randn(1, 1, 224, 224)
    traced = torch.jit.trace(model, dummy_heatmap)
    traced.save(path)

