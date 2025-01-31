# Multi-View CNN
1. Render different viewpoints
2. extract image features with 2D CNN
3. element wise max pooling across views
4. another CNN for classification

* Good performance
* Can leverage vast literature of image classification
* Can use pretrained features
* Not efficient if we need to render and process a multitude of different viewpoints
* What if the input is noisy and/or incomplete? E.g. point cloud from only one viewpoint

# 3D CNN on Volumetric Data

Complexity issue

**Solution** levarage sparsity of 3D Shapes
$\Longrightarrow$  use Octree (OctNet)

# PointNet

End-to-end learning for scattered, unordered, sparse, noisy point data
Unified framework for various tasks: 
* Classification
* Part Segmentation
* Semantic Segmentation

## Challange 1: permuation invarince
Input points are unordered, the model needs to be invariant to $N!$ permutations. 

### Symmetric Function
![[Pasted image 20250131102544.png]]
$h, \gamma$ can be **MLP** and $g$ **max pooling**
This would be the (vanilla) **PointNet** 


## Challange 2: invariance under geometric transformations
Point cloud rotation should not alter classification results. 

### Input Alignment by Transformer Network
A sub-module learns to predict a 3D transformation to align the input. 
![[Pasted image 20250131103011.png]]


## PointNet **Classification** Network 

![[Pasted image 20250131103346.png]]
1. Transform 3D points
2. Pass points individually through a MLP (shared weights) to take them to higher dimensional space (64). 
3. Apply another transform but this time in 64 dimensional space
4. Again pass through a MLP to increase dimensionality
5. Symmetric operation: Max pooling over channel dimension
6. Pass the global feature to a final MLP outputting the logits


## PointNet **Segmentation** Network 

![[Pasted image 20250131104906.png]]
Concatenate local embedding and global feature. Then pass points through shared MLP, taking them down to 128 dimensional space. And finally map every point to class logits. 

## Limitations of Point Net

Does global feature learning, meaning that features in the network are not local, they are for **one** point or **totally global**.  

Global feature depends on absolute coordinates. 

# PointNet++ 
Tries to mimic convolution (when pixels are in a grid, we can easily define a neighborhood).
Points clouds do not have this regular pixel pattern. 

Solutions: 
* Ball query (results in more stable, fixed "kernel" size)
* K nearest neighbor (K-NN) query

PointNet++ uses the Ball query.

### Set abstraction
The set abstraction level is made of three key layers: 
* sampling layer 
* grouping layer
* PointNet layer.

![[Pasted image 20250131114253.png]]
#### Sampling layer 
Given input points $\{x_1, x_2, \ldots, x_n\}$, **iterative farthest point sampling** is used to choose a subset of points $\{x_{i_1}, x_{i_2}, \ldots, x_{i_m}\}$, such that $x_{i_j}$ is the most distant point (in metric distance) from the set $\{x_{i_1}, x_{i_2}, \ldots, x_{i_{j-1}}\}$ with regard to the rest points.

Compared with random sampling, it has **better coverage** of the entire point set.

#### Grouping layer
The input to this layer is a point set of size $N \times (d + C)$ and the coordinates of a set of centroids of size $N' \times d$. The output are groups of point sets of size $N' \times K \times (d + C)$, where each group corresponds to a local region and $K$ is the number of points in the neighborhood of centroid points.

- $K$ varies across groups but the succeeding PointNet layer is able to convert a flexible number of points into a fixed-length local region feature vector.
- Ball query finds all points that are within a radius to the query point.

#### PointNet layer
1. individual point MLP 
2. max pooling across channel dimension
3. MLP on global channel 

The point net layer is applied to the groups from the groping layer. 
There is only one set of parameters for the PointNet layer, every group goes through the same PointNet (shared weights). 

## Inference
![[Pasted image 20250131120323.png]]
For segmentation we need to go from the sparse feature to a dense prediction. 
Hence we need to interpolate. The unit pointnet is just a MLP, through which each point is passed individually.  

The interpolation is linear from the K-NN (K = 3). Where the neighbors are weighted by the inverse of their 3D euclidian distance. 


# LiDAR (unimodal) 3D object detection
The provide reliable depth information, however unlike images, LiDAR pcds are sparse, irregular and have highly variable point density. 

## VoxelNet: object proposals from voxels

### Voxelization
Helps with: 
* computation savings
* decreases the imbalance in point density

It works by: 
1. Group point cloud (into voxels)
2. Sample t points from every group
3. For every point augment its 3D position with the offset from the centroid of the other sampled points in its group and the received radiant power. 
4. pass the point vector through a simple MLP, into a feature space
5. do max pooling in the channel dimension over all the point from a voxel to produce a single voxel feature 

This gives a sparse 4D tensor representing the point cloud. This tensor is passed to the convolutional middle layer.

### Convolutional middle layer
A **convolutional middle layer** applies 3D convolution, Batch Norm (BN) and ReLU. 
They aggregate voxel-wise features within a expanding receptive field, adding more context. 

They progressively consolidate the vertical 3D dimension to only bird's eye view (BEW) (2 spatial dimensions).  
The BEW feature map is passed to the **region proposal network** (RPN). 

### Region proposal network 
Is essentially the same [[6 - Object Detection#RoI Pooling]] as in [[6 - Object Detection#Fast R-CNN

A typical [[4 - Semantic Segmentation#U-Net]] style encoder-decoder network. 

The RoI are uppsampled and aggregated with skip connections and then one branch predicts RoI class probabilities and another branch does **bounding box regression**. 
![[Pasted image 20250131171932.png]]

### Bounding box regression
We parameterize a 3D ground truth box as  $(x_c^g, y_c^g, z_c^g, l^g, w^g, h^g, \theta^g)$ 
where $x_c^g, y_c^g, z_c^g$ represent the center location, $l^g, w^g, h^g$ are length, width, height of the box, and $\theta^g$ is the yaw rotation around the Z-axis.

By comparing  $(x_c^a, y_c^a, z_c^a, l^a, w^a, h^a, \theta^a)$  to ground truth, we define the residual vector  $\mathbf{u}^* \in \mathbb{R}^7$ 
containing 7 regression targets corresponding to center locations $\Delta x, \Delta y, \Delta z$, three dimensions $\Delta l, \Delta w, \Delta h$, and the rotation $\Delta \theta$, which are computed as:
$$
\Delta x = \frac{x_c^g - x_c^a}{d_a}, \quad 
\Delta y = \frac{y_c^g - y_c^a}{d_a}, \quad 
\Delta z = \frac{z_c^g - z_c^a}{h_a},$$ $$
\Delta l = \log\left(\frac{l^g}{l^a}\right), \quad 
\Delta w = \log\left(\frac{w^g}{w^a}\right), \quad 
\Delta h = \log\left(\frac{h^g}{h^a}\right), \quad 
\Delta \theta = \theta^g - \theta^a.
$$

Diagonal of the base of the anchor box: $d_a = \sqrt{(l^a)^2 + (w^a)^2}.$

The **loss function** is defined as:
$$
L = \alpha \frac{1}{N_{\text{pos}}} \sum_i L_{\text{cls}}(p_i^{\text{pos}}, 1) 
+ \beta \frac{1}{N_{\text{neg}}} \sum_j L_{\text{cls}}(p_j^{\text{neg}}, 0) 
+ \frac{1}{N_{\text{pos}}} \sum_i L_{\text{reg}}(\mathbf{u}_i, \mathbf{u}_i^*).
$$

VoxelNet uses a multi-task loss similar to that employed by [[6 - Object Detection#Faster R-CNN]] for region proposal generation:
- Smooth L1 loss for regression only for positive anchors
- Cross-entropy for classification into positive and negative anchors for both types.
