# VoxelMap

**[Updated] Date of release**: We have just received the reviewer comments in the first round of paper reviews and the source code and dataset will be released by the middle of June.
## Introduction
**VoxelMap** is an efficient and probabilistic adaptive(coarse-to-fine) voxel mapping method for 3D LiDAR. Unlike the point cloud map, VoxelMap uses planes as representation units. A scan of LiDAR data will generate or update the plane. Each plane contains its own plane parameters and uncertainties that need to be estimated. This repo shows how to integrate VoxelMap into a LiDAR(-Inertial) odometry.

<div align="center">
    <img src="pics/kitti_mapping.png" width = 100% >
    <font color=#a0a0a0 size=2>The plane map constructed by VoxelMap on KITTI Odometry sequence 00.</font>
</div>
  

<div align="center">
    <img src="pics/park_mapping.png" width = 100% >
    <font color=#a0a0a0 size=2>The plane map constructed by VoxelMap in the park environment with Livox Avia LiDAR.</font>
</div>

<div align="center">
    <img src="pics/mountain_mapping.png" width = 100% >
    <font color=#a0a0a0 size=2>The plane map constructed by VoxelMap in the mountain environment with Livox Avia LiDAR.</font>
</div>

### Developers:
[Chongjian Yuan 袁崇健](https://github.com/ChongjianYUAN)， [Wei Xu 徐威](https://github.com/XW-HKU)


### Related paper
Related paper available on **arxiv**:  
[Efficient and Probabilistic Adaptive Voxel Mapping for Accurate Online LiDAR Odometry](https://arxiv.org/abs/2109.07082)

### Related video
Our accompanying videos are now available on **YouTube**.
<div align="center">
    <a href="https://youtu.be/HSwQdXg31WM" target="_blank">
    <img src="pics/video_cover.png" width=60% />
</div>
