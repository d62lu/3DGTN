import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def sample_and_group_all_gt(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = xyz
    grouped_xyz = xyz.view(B, 1, N, C)

    relative_coord = grouped_xyz[:, :, None, :, :] - grouped_xyz[:,:, :, None, :]

    pos_dist_sqrt=relative_coord

    if points is not None:
        new_points =points.view(B, 1, N, -1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, pos_dist_sqrt

class PatchMerging_avepooling(nn.Module):
    def __init__(self, n):
        super().__init__()

        self.avepooling_layer=nn.AdaptiveAvgPool2d((n,1))
        ####################################################

    def forward(self, x):

        x = x.permute(0, 3, 1, 2)
        x = self.avepooling_layer(x)
        x = x.permute(0, 2, 3, 1)
        
        return x


class WindowAttention(nn.Module):
    def __init__(self, input_dim, output_dim, heads, head_dim, N, attn_drop_value):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.N = N
        self.input_dim=input_dim 
        self.output_dim=output_dim

        self.linear_input = nn.Sequential(nn.Linear(input_dim, output_dim), nn.BatchNorm2d(output_dim), nn.ReLU(inplace=True))
        self.linear_out = nn.Sequential(nn.BatchNorm2d(output_dim), nn.ReLU(inplace=True))

        self.q_conv = nn.Conv2d(input_dim, output_dim, 1)
        self.k_conv = nn.Conv2d(input_dim, output_dim, 1)

        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm3d(3), nn.ReLU(inplace=True), nn.Linear(3, 1))

        self.v_conv = nn.Conv2d(input_dim, output_dim, 1)

        self.trans_conv = nn.Conv2d(output_dim, output_dim, 1)
        self.after_norm = nn.BatchNorm2d(output_dim)
        self.act = nn.ReLU()

        ####################################channel-wise CAA##################
        self.bn1 = nn.BatchNorm1d(N//8)
        self.bn2 = nn.BatchNorm1d(N//8)
        self.bn3 = nn.BatchNorm1d(input_dim)

        self.query_conv = nn.Sequential(nn.Conv1d(in_channels=N, out_channels=N//8, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv1d(in_channels=N, out_channels=N//8, kernel_size=1, bias=False),
                                        self.bn2,
                                        nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=1, bias=False),
                                        self.bn3,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1))

        ####################################channel-wise CAA##################

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, p_r):


        

        b, n, K, _, h = *x.shape, self.heads

        for i, layer in enumerate(self.linear_input): x = layer(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) if i == 1 else layer(x)

        
        x_q = self.q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        x_k = self.k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_v = self.v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # b, n, k, c


        x_q = rearrange(x_q, 'b n k (h d) -> b h n k d',
                        h=h, k=K)
        x_k = rearrange(x_k, 'b n k (h d) -> b h n k d',
                        h=h, k=K)
        x_v = rearrange(x_v, 'b n k (h d) -> b h n k d',
                        h=h, k=K)

        # b,h,n,k,d

        dots = einsum('b h n i d, b h n j d -> b h n i j', x_q, x_k)* self.scale 
        # b,h,n,k,k

        for i, layer in enumerate(self.linear_p): p_r = layer(p_r.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1) if i == 1 else layer(p_r)
        # b,n,k,k,1

        p_r=p_r.view(b,n,K,-1)
        # b,n,k,k

        p_r=p_r.unsqueeze(-1)
        # b,n,k,k, 1

        p_r=p_r.permute(0,4,1,2,3)

        # b,1,n,k,k
        dots=dots+p_r


         ########################################standard style##############################################
        attention = self.softmax(dots)
        attention = attention / (1e-9 + attention.sum(dim=-1, keepdim=True)) # point cloud transformer

        x_hat = x.squeeze(1) # ( B X N X C )
        shortcut=x_hat.permute(0, 2, 1) # ( B X C X N )
        proj_query = self.query_conv(x_hat) 
        proj_key = self.key_conv(x_hat).permute(0, 2, 1) 
        similarity_mat = torch.bmm(proj_key, proj_query) # ( B X C X C )

        # Channel Affinity Estimator block
        affinity_mat = torch.max(similarity_mat, -1, keepdim=True)[0].expand_as(similarity_mat)-similarity_mat
        affinity_mat = self.softmax(affinity_mat) # ( B X C X C )
        
        proj_value = self.value_conv(x_hat.permute(0, 2, 1)) # ( B X C X N )
        out = torch.bmm(affinity_mat, proj_value) # ( B X C X N )
        # residual connection with a learnable weight
        out = self.alpha*out + shortcut 

        out = out.permute(0, 2, 1) # ( B X N X C )

       

        return out, attention
        # return  attention

class SwinBlock(nn.Module):
    def __init__(self, batchnorm, input_dim, output_dim, heads, head_dim, mlp_dim, patch_size, attn_drop_value, feed_drop_value):
        super().__init__()
        self.attention_block = WindowAttention(input_dim=input_dim, 
                                                output_dim=output_dim,
                                                heads=heads,
                                                head_dim=head_dim,
                                                N=patch_size,
                                                attn_drop_value=attn_drop_value)

    def forward(self, x, pos_dist_sqrt):
        x = self.attention_block(x, pos_dist_sqrt)
        return x

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)), inplace=True)

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class Global_Transformer(nn.Module):
    #def __init__(self, npoint, radius, nsample, in_channel, out_channels, mlp, group_all, layers, num_heads, head_dim):
    def __init__(self, avepooling, batchnorm, attn_drop_value, feed_drop_value, npoint, in_channel, out_channels, layers, num_heads, head_dim):
        super(Global_Transformer, self).__init__()
        self.npoint = npoint
        self.avepooling = avepooling
        
        self.layer = SwinBlock(batchnorm, input_dim=in_channel, output_dim=out_channels, heads=num_heads, head_dim=head_dim, mlp_dim=in_channel * 4,
                          patch_size=npoint, attn_drop_value=attn_drop_value, feed_drop_value=feed_drop_value)

        self.linear_out = nn.Sequential(nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True))

        self.trans_conv = nn.Conv1d(out_channels, out_channels, 1)
        self.after_norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        #################################################################################################

    def forward(self, xyz, points, grouped_points_list):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
            grouped_points_list: [B, D, K, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        
        
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        shortcut = points # b n d 

        new_xyz, new_points, pos_dist_sqrt = sample_and_group_all_gt(xyz, points)
        # new_xyz: sampled points position data, [B, npoint, 3]
        # new_points: sampled points data, [B, 1, npoint, D]
        B, S, _ = new_xyz.shape
        ######################################################################
        
 
        global_points, attention_map=self.layer(new_points, pos_dist_sqrt) # b,h,1,n, n

        _,h,_,_,_ = attention_map.shape
        attention_map = attention_map.squeeze(2) #b h n n


        local_points_list =[]
        for i, grouped_points in enumerate(grouped_points_list):
            grouped_points = grouped_points.permute(0,3,2,1) #[B, N, K, D]
            _,_,K,C = grouped_points.shape


            # grouped_points = rearrange(grouped_points, 'b n k (h d) -> b h n k d',
            #             h=h, b=B, k=K, n=S) # b h n k d

            # grouped_points = rearrange(grouped_points, 'b h n k d -> b h n (kd)',
            #             h=h, b=B, k=K, n=S) # b h n kd

            grouped_points = rearrange(grouped_points, 'b n k (h d) -> b h n (k d)',
                        b=B, h=h,  k=K, n=S) # b h n k d

            
            out = einsum('b h n n, b h n d -> b h n d', attention_map, grouped_points)
            # b,h,n,kd

            out = rearrange(out, 'b h n (k d) -> b h n k d',
                        h=h, b=B, k=K, n=S) # b h n k d
            
            out = rearrange(out, 'b h n k d -> b n k (h d)',
                        h=h, b=B, k=K, n=S) # b n k D




            local_points = torch.max(out, 2)[0]  # [B, N, D]
            local_points_list.append(local_points)

        local_points_concat = torch.cat(local_points_list, dim=-1)# [B, N, D]      
        
        #####################################################################
        output_points = local_points_concat + global_points  # [B, N, D]
        output_points = shortcut.permute(0, 2, 1) + self.act(self.after_norm(self.trans_conv(output_points.permute(0, 2, 1)))) # b d n 
        #####################################################################
        
        new_xyz = new_xyz.permute(0, 2, 1)
        
        return new_xyz, output_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = 2*in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        _, _, D = points.shape
        S = self.npoint

        fps_index=farthest_point_sample(xyz, S)

        new_xyz = index_points(xyz, fps_index)
        new_points_ori = index_points(points, fps_index) # B S D

        new_points_list = []
        grouped_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)

            grouped_xyz = grouped_xyz/radius # (pi - pj)/r, pointnext

            if points is not None:
                grouped_points = index_points(points, group_idx)
                new_points_ori_K=new_points_ori
                new_points_ori_K = new_points_ori_K.contiguous().view(B, S, 1, D).repeat(1, 1, K, 1)
                grouped_points=torch.cat((grouped_points-new_points_ori_K, new_points_ori_K), dim=-1).contiguous()
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)), inplace=True)
            grouped_points_list.append(grouped_points) # [B, D, K, S]
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat, grouped_points_list


class InputEmbedding(nn.Module):
    def __init__(self, radius_list, nsample_list, in_channel, mlp_list):
        super(InputEmbedding, self).__init__()
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = 2*in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        _, _, D = points.shape


        new_xyz = xyz
        new_points_ori = points

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, N, 1, C)

            grouped_xyz = grouped_xyz/radius # (pi - pj)/r, pointnext

            if points is not None:
                grouped_points = index_points(points, group_idx)
                new_points_ori_K=new_points_ori
                new_points_ori_K = new_points_ori_K.contiguous().view(B, N, 1, D).repeat(1, 1, K, 1)
                grouped_points=torch.cat((grouped_points-new_points_ori_K, new_points_ori_K), dim=-1).contiguous()
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)), inplace=True)
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)), inplace=True)
        return new_points

