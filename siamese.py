import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, feature_dim=256):
        super(SiameseNetwork, self).__init__()
        self.embedding_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, input1, input2):
        out1 = self.embedding_net(input1)
        out2 = self.embedding_net(input2)
        return out1, out2

    def compute_similarity(self, out1, out2, metric='euclidean'):
        if metric == 'euclidean':
            return F.pairwise_distance(out1, out2)
        elif metric == 'cosine':
            return 1 - F.cosine_similarity(out1, out2)
        else:
            raise ValueError("Unsupported metric")
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # label = 0 if similar, 1 if dissimilar
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = (1 - label) * torch.pow(euclidean_distance, 2) + \
               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()
def training_step(model, loss_fn, optimizer, visible_feat, heatmap_feat, label):
    model.train()
    output1, output2 = model(visible_feat, heatmap_feat)
    loss = loss_fn(output1, output2, label)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
def get_similarity_score(model, feat1, feat2, metric='cosine'):
    model.eval()
    with torch.no_grad():
        emb1, emb2 = model(feat1, feat2)
        score = model.compute_similarity(emb1, emb2, metric)
        return score.item()
