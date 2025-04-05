from torchvision import transforms
import random
from PIL import ImageDraw

class OcclusionSimulator:
    """
    Simulates face occlusions like sunglasses, scarves, or masks using random rectangles.
    Useful during training to improve robustness.
    """
    def __init__(self, probability=0.5):
        self.probability = probability

    def apply_occlusion(self, image):
        if random.random() > self.probability:
            return image

        draw = ImageDraw.Draw(image)
        w, h = image.size
        num_boxes = random.randint(1, 3)

        for _ in range(num_boxes):
            x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
            x2, y2 = random.randint(w//2, w), random.randint(h//2, h)
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

        return image
class AdvancedImagePreprocessor:
    def __init__(self, input_size=(224, 224), simulate_occlusion=False):
        self.simulate_occlusion = simulate_occlusion
        self.occluder = OcclusionSimulator(probability=0.7)

        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
        ])

    def preprocess(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            if self.simulate_occlusion:
                image = self.occluder.apply_occlusion(image)
            return self.transform(image).unsqueeze(0)
        except Exception as e:
            logging.error(f"Error in preprocessing: {e}")
            return None
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale
class ResNetSE(nn.Module):
    def __init__(self, base_model):
        super(ResNetSE, self).__init__()
        self.base_layers = list(base_model.children())
        self.conv1 = self.base_layers[0]
        self.bn1 = self.base_layers[1]
        self.relu = self.base_layers[2]
        self.maxpool = self.base_layers[3]
        self.layer1 = self._add_se(self.base_layers[4])
        self.layer2 = self._add_se(self.base_layers[5])
        self.layer3 = self._add_se(self.base_layers[6])
        self.layer4 = self._add_se(self.base_layers[7])
        self.avgpool = self.base_layers[8]
        self.embedding = nn.Linear(base_model.fc.in_features, 512)

    def _add_se(self, layer):
        for i in range(len(layer)):
            layer[i].add_module('se', SEBlock(layer[i].conv1.out_channels))
        return layer

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return self.embedding(x)
# Export to ONNX
def export_model_to_onnx(model, path="visible_feature_extractor.onnx"):
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, path, input_names=["input"], output_names=["features"])
    logging.info(f"Model exported")

# Export to TorchScript
def export_model_to_torchscript(model, path="visible_feature_extractor.pt"):
    scripted_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
    scripted_model.save(path)
    logging.info(f"Model saved as TorchScript")
