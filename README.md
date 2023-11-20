# 3DGTN
3DGTN: 3D Dual-Attention GLocal Transformer Network for Point Cloud Classification and Segmentation

This is a Pytorch implementation of 3DGTN.

Paper link: https://arxiv.org/abs/2209.11255

# Abstract

Although the application of Transformers in 3D point cloud processing has achieved significant progress and success, it is still challenging for existing 3D Transformer methods to efficiently and accurately learn both valuable global features and valuable local features for improved applications. This paper presents a novel point cloud representational learning network, called 3D Dual Self-attention Global Local (GLocal) Transformer Network (3DGTN), for improved feature learning in both classification and segmentation tasks, with the following key contributions. First, a GLocal Feature Learning (GFL) block with the dual self-attention mechanism (i.e., a novel Point-Patch Self-Attention, called PPSA, and a channel-wise self-attention) is designed to efficiently learn the GLocal context information. Second, the GFL block is integrated with a multiscale Graph Convolution-based Local Feature Aggregation (LFA) block, leading to a Global-Local (GLocal) information extraction module that can efficiently capture critical information. Third, a series of GLocal modules are used to construct a new hierarchical encoder-decoder structure to enable the learning of ”GLocal” information in different scales in a hierarchical manner. The proposed framework is evaluated on both classification and segmentation datasets, demonstrating that the proposed method is capable of outperforming many state-of-the-art methods on both synthetic and LiDAR data.

# Architecture

<img width="739" alt="1700503852490" src="https://github.com/d62lu/3DGTN/assets/92398834/3362ebdc-6502-4661-a327-ed99f3739550">

# Install
The latest codes are tested on CUDA10.1, PyTorch 1.6, and Python 3.8.

# Data preparation
Download the alignment ModelNet (https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save it in "data/modelnet40_normal_resampled/".

Download the ShapeNet dataset (https://shapenet.org/) and save it in "data/shapenetcore_partanno_segmentation_benchmark_v0_normal/".


# Run

## For classification
```
python train_classification.py --use_normals --log_dir 3dgtn_cls --process_data
```

## For segmentation
```
python train_seg.py --normal --log_dir 3dgtn_seg
```




