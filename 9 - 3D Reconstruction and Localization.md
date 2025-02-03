
# Iterative Closest Point (ICP)
**Optimization problem** statement: Given two corresponding sets of 3D points:
$$
\mathcal{P}_s = \begin{bmatrix} \mathbf{P}_s^{(1)} & \mathbf{P}_s^{(2)} & \cdots & \mathbf{P}_s^{(n)} \end{bmatrix}
$$
$$
\mathcal{P}_{s'} = \begin{bmatrix} \mathbf{P}_{s'}^{(1)} & \mathbf{P}_{s'}^{(2)} & \cdots & \mathbf{P}_{s'}^{(n)} \end{bmatrix}
$$

Obtain 3D Euclidean motion $\{\hat{\mathbf{R}}, \hat{\mathbf{t}}\}$ and assignment function $\hat{\phi}$ that minimize the sum of squared errors:
$$
E(\mathbf{R}, \mathbf{t}, \phi) = \sum_{i=1}^n \left\lVert \mathbf{R} \mathbf{P}_s^{(\phi(i))} + \mathbf{t} - \mathbf{P}_{s'}^{(i)} \right\rVert_2^2
$$
where $\phi: \{1, \dots, n\} \to \{1, \dots, n\}$ and $\mathbf{P}_s^{(\phi(i))}$ and $\mathbf{P}_{s'}^{(i)}$ are corresponding points induced by the assignment function $\phi$.

When the point correspondences are known, the translation and rotation can be calculated in closed form.

For ICP we assume that points with minimal distance correspond. Hence good initialization is needed. 

# Visual Odometry

![[Pasted image 20250202142223.png]]
## RANSAC for robust SfM
![[Pasted image 20250202142431.png]]

## Feature Matching in VO
There is the traditional descriptor matching (e.g. SIFT) or KLT tracking 
and there is **learned feature matching**.

**Motivation for learned matching**: 
- Nearest neighbor matching ignores the assignment structure and **discards spatial information about the arrangement of keypoints** within each frame of the pair.
- 2D feature keypoints are projections of salient 3D scene points, such as corners or blobs. Thus, feature matches across the two frames must adhere to certain physical constraints:
	  1. A keypoint can have **at most a single correspondence** in the other image.
	  2. Some keypoints will be **unmatched** due to occlusion and failure of the detector.
	- Thus, an effective model for feature matching should aim at:
	  1. Finding all correspondences between reprojections of the same 3D points.
	  2. Identifying keypoints that have no matches.
- The two above objectives can be formulated jointly as a **differentiable global optimization problem** in order to provide a global solution end-to-end, in contrast to local, per-keypoint nearest neighbor matching.

## SuperGlue
Learned feature matching. 

### Attentional context aggregation
![[Pasted image 20250202151804.png]]
Start with "traditional" descriptors (e.g. SIFT) from both images. 
Add a positional embedding and pass every feature plus embedding through a small keypoint encoder MLP. 

Then pass the features through $L$ blocks of **Attentional Aggregation**. 
One block first has a **self-attention** and then a **cross-attention** layer. 

* in self-attention features attend to all the other features from the same image
![[Pasted image 20250202151835.png]]

* in cross-attention each feature attends to all the features from the other frame
![[Pasted image 20250202151856.png]]

### Differentiable optimal matching
The attentional aggregated features are than passed to the matching. 

Start with simple dot product scores and dustbin for unmatched features: ![[Pasted image 20250202153426.png]]
The initial dot product scores are not normalized, this is done using the **Sinkhorn Algorightm**. It alternates the nomalization of rows and columns of the exponentiated score matrix.
This is differentiable and can hence be used to backpropagate the gradient loss to the **Attentional Aggregation** stage.

The loss used is $L_{matching} = -\sum_{(i, j) \in M}log(P_{ij}$) where $M$ are the ground truth matches. 


# 3D represenations

The classics: ![[Pasted image 20250202143025.png]]
These have some issues: 
- **Point clouds, meshes, voxel grids**  
  - 3D  
  - Discrete representations  
  - Information loss  
- **Multi-view images**  
  - High resolution  
  - Reduce the dimensionality of the scene  
- **Implicit functions**  
  - Volumetric 3D representations  
  - Typically need access to ground-truth geometry  

## Neural radiance fields (NeRF) 

MLP mapping **3D position, 2D direction $\rightarrow$ density at 3D position and RGB color**


![[Pasted image 20250202143335.png]]

- Representation of the scene via a continuous 5D implicit function implemented via an MLP.
- Mapping each **3D position + 2D viewing direction** to a volume **density** $\sigma$ and a view-dependent RGB color/**radiance**.
- These outputs are used to render images of the scene through classical volume rendering.
- Learning the MLP via image-level supervision thanks to differentiability of this pipeline.

### Architecture 
![[Pasted image 20250203094032.png]]
The density $\sigma$ at $x$ does not depend on the viewing direction $(\theta, \phi)$, hence the first MLP predicts a latent vector $f$ and $\sigma$ and then $(\theta, \phi)$ is appended to $f$ and passed to the second MLP, which predicts the color/radiance. 

### Volume rendering loss
![[Pasted image 20250203093752.png]]

Discrete approximation 
![[Pasted image 20250203093825.png]]

The color $\hat{C}$ is differentiable and hence the difference to ground truth color from training images is used as a loss. 

### Positional encoding 
Deep networks are biased towards learning lower-frequency representations, meaning that the  scene will miss details and is not sharp. 

![[Pasted image 20250203094549.png]]

![[Pasted image 20250203094524.png]]

### Limitations 
* The scene has limited dimensions in 3D space
* Strong assumption of a static scene: no changes in illumination or medium density, no transient/dynamic objects


## NeRF in the wild: image-dependent embeddings
![[Pasted image 20250203100435.png]]


## Block-NeRF: neural driving scene 3D reconstruction at scale

Multiple NeRF make up one scene. (One NeRF at every intersection)
![[Pasted image 20250203101120.png]]

Add a visibility map predictor to each NeRF. 
![[Pasted image 20250203101208.png]]

This visibility map is then used to compose the NeRF outputs. 

![[Pasted image 20250203101308.png]]