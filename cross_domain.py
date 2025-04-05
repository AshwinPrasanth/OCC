class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, output_dim=128):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, H, W]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 32, H/2, W/2]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, H/2, W/2]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [B, 64, H/4, W/4]
        )

    def forward(self, x):
        return self.features(x)  # return intermediate feature maps
class EarlyFusionNet(nn.Module):
    def __init__(self, visible_in_channels=3, heatmap_in_channels=1):
        super(EarlyFusionNet, self).__init__()
        self.visible_cnn = SimpleCNN(in_channels=visible_in_channels)
        self.heatmap_cnn = SimpleCNN(in_channels=heatmap_in_channels)

        self.fusion_net = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Fuse features
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling → [B, 64, 1, 1]
            nn.Flatten(),                 # → [B, 64]
            nn.Linear(64, 32),            # Final embedding
            nn.ReLU()
        )

    def forward(self, visible_img, heatmap_img):
        vis_feat = self.visible_cnn(visible_img)      # [B, 64, H/4, W/4]
        heat_feat = self.heatmap_cnn(heatmap_img)     # [B, 64, H/4, W/4]
        combined_feat = torch.cat((vis_feat, heat_feat), dim=1)  # Channel-wise concat
        output = self.fusion_net(combined_feat)       # → [B, 32]
        return output

