import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import InputEmbedding, PointNetSetAbstractionMsg, PointNetSetAbstraction, Global_Transformer
import torch

class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel


        self.embedding = InputEmbedding([0.1], [16], in_channel,[[32, 32, 64]])
        

        self.sa1 = PointNetSetAbstractionMsg(256, [0.1, 0.2, 0.4], [16, 32, 128], 64,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.global_sa1 = Global_Transformer(avepooling=False, batchnorm=True, attn_drop_value=0, feed_drop_value=0, npoint=256, in_channel=320, out_channels=320, layers=1, num_heads=8, head_dim=40)
        self.sa2 = PointNetSetAbstractionMsg(64, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.global_sa2 = Global_Transformer(avepooling=False, batchnorm=True, attn_drop_value=0, feed_drop_value=0, npoint=64, in_channel=640, out_channels=640, layers=1, num_heads=16, head_dim=40)
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.fc1 = nn.Linear(1984, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        xyz, embeded_points = self.embedding(xyz, norm)

        l1_xyz, l1_points, l1_grouped_points_list = self.sa1(xyz, embeded_points)
        g1_xyz, g1_points = self.global_sa1(l1_xyz, l1_points, l1_grouped_points_list) # b d n

        global_feats1 = self.alpha*torch.max(g1_points, 2)[0] # b d

        l2_xyz, l2_points, l2_grouped_points_list = self.sa2(g1_xyz, g1_points)
        g2_xyz, g2_points = self.global_sa2(l2_xyz, l2_points, l2_grouped_points_list) # b d n

        global_feats2 = self.beta*torch.max(g2_points, 2)[0]# b d

        l3_xyz, l3_points = self.sa3(g2_xyz, g2_points)

        global_feats3 = l3_points.view(B, 1024)

        x = torch.cat((global_feats1, global_feats2, global_feats3), dim=-1) # 320+640+1024 
        x = self.drop1(F.relu(self.bn1(self.fc1(x)), inplace=True))
        x = self.drop2(F.relu(self.bn2(self.fc2(x)), inplace=True))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, gold, smoothing=True):
        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss

