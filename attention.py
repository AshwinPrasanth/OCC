class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 8),
            nn.ReLU(),
            nn.Linear(in_channels // 8, in_channels),
            nn.Sigmoid()
        )
        
        # Spatial Attention
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        channel_att = self.fc(avg_out).view(b, c, 1, 1)
        x = x * channel_att

        # Spatial Attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        mean_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, mean_pool], dim=1)
        spatial_att = self.conv(spatial_input)
        x = x * spatial_att

        return x
class AttentionFusionNet(nn.Module):
    def __init__(self):
        super(AttentionFusionNet, self).__init__()
        self.visible_cnn = SimpleCNN(in_channels=3)
        self.heatmap_cnn = SimpleCNN(in_channels=1)
        self.attention = AttentionModule(in_channels=128)

        self.fusion_net = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, visible_img, heatmap_img):
        vis_feat = self.visible_cnn(visible_img)      # [B, 64, H, W]
        heat_feat = self.heatmap_cnn(heatmap_img)     # [B, 64, H, W]
        fused = torch.cat((vis_feat, heat_feat), dim=1)  # [B, 128, H, W]
        attended = self.attention(fused)              # Apply attention
        out = self.fusion_net(attended)
        return out
from torchvision.models import resnet18

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNetFeatureExtractor, self).__init__()
        base_model = resnet18(pretrained=False)
        if in_channels == 1:
            base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove final FC layers
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])  # [B, 512, H/32, W/32]

    def forward(self, x):
        return self.feature_extractor(x)
class ResNetFusionNet(nn.Module):
    def __init__(self):
        super(ResNetFusionNet, self).__init__()
        self.resnet_visible = ResNetFeatureExtractor(in_channels=3)
        self.resnet_heatmap = ResNetFeatureExtractor(in_channels=1)
        
        self.attention = AttentionModule(in_channels=1024)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )

    def forward(self, visible_img, heatmap_img):
        vis_feat = self.resnet_visible(visible_img)    # [B, 512, H, W]
        heat_feat = self.resnet_heatmap(heatmap_img)   # [B, 512, H, W]
        fused = torch.cat((vis_feat, heat_feat), dim=1)  # [B, 1024, H, W]
        attended = self.attention(fused)               # Apply attention
        out = self.classifier(attended)
        return out
