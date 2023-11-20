import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetFeaturePropagation, InputEmbedding, PointNetSetAbstractionMsg, PointNetSetAbstraction, Global_Transformer
import torch

class get_model(nn.Module):
    def __init__(self,num_classes,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        embeded_channel = 64
        cls_class = 16

        self.embedding = InputEmbedding([0.1], [32], in_channel,[[32, 32, embeded_channel]])

        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 64,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.global_sa1 = Global_Transformer(avepooling=False, batchnorm=True, attn_drop_value=0, feed_drop_value=0, npoint=512, in_channel=320, out_channels=320, layers=1, num_heads=8, head_dim=40)
        
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.global_sa2 = Global_Transformer(avepooling=False, batchnorm=True, attn_drop_value=0, feed_drop_value=0, npoint=128, in_channel=640, out_channels=640, layers=1, num_heads=16, head_dim=40)

        self.sa3 = PointNetSetAbstractionMsg(32, [0.4, 0.8, 1], [16, 32, 64], 640,[[128, 128, 256], [256, 256, 512], [256, 256, 512]])
        self.global_sa3 = Global_Transformer(avepooling=False, batchnorm=True, attn_drop_value=0, feed_drop_value=0, npoint=32, in_channel=1280, out_channels=1280, layers=1, num_heads=32, head_dim=40)
        
        self.sa4 = PointNetSetAbstraction(None, None, None, 1280 + 3, [256, 512, 1024], True)

        self.fp4 = PointNetFeaturePropagation(in_channel=1024+320+640+1280+(1280), mlp=[1024, 512])
        
        self.fp3 = PointNetFeaturePropagation(in_channel=512+640, mlp=[512, 256])

        self.fp2 = PointNetFeaturePropagation(in_channel=256+320, mlp=[256, 128])

        self.fp1 = PointNetFeaturePropagation(in_channel=128+cls_class+3+embeded_channel, mlp=[128, 128])
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.gama = nn.Parameter(torch.zeros(1))

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        B, _, N = xyz.shape
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

        l3_xyz, l3_points, l3_grouped_points_list = self.sa3(g2_xyz, g2_points)
        g3_xyz, g3_points = self.global_sa3(l3_xyz, l3_points, l3_grouped_points_list) # b d n
        global_feats3 = self.gama*torch.max(g3_points, 2)[0]# b d

        l4_xyz, l4_points = self.sa4(g3_xyz, g3_points)
        global_feats4 = l4_points.view(B, 1024)

        x = torch.cat((global_feats1, global_feats2, global_feats3, global_feats4), dim=-1) # 320+640+1024    B D
        x = x.unsqueeze(-1)

        # Feature Propagation layers
        l3_points = self.fp4(g3_xyz, l4_xyz, g3_points, x)

        l2_points = self.fp3(g2_xyz, l3_xyz, g2_points, l3_points)

        l1_points = self.fp2(g1_xyz, g2_xyz, g1_points, l2_points)

        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)

        l0_points = self.fp1(xyz, g1_xyz, torch.cat([cls_label_one_hot,xyz,embeded_points],1), l1_points)
       
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

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

