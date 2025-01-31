# Monocular Depth Estimation
An ill-posed problem: a single 2D image may be produced from an infinite number of distinct 3D scenes

Standard regression problem, using mean squared error (MSE) loss. 
## Motivation
* Existing monocular data 
* Physical limit (e.g. no space for two cameras)
* Cheap and easier to maintain

## Metrics 
Evaluating a predicted depth map **Y** and its ground truth depth image **Y\*** with **T** depth pixels.
### Threshold
The percentage of predictions \( y \) such that:
$$
\max\left(\frac{y_i}{y_i^*}, \frac{y_i^*}{y_i}\right) = \sigma < thr
$$
$\sigma = 1.25^i$ with $i \in \{1, 2, 3\}$ is common choice. 
### Absolute Relative Difference
The absolute relative difference is calculated as:
$$
\text{rel} = \frac{1}{T} \sum_{i,j} \frac{|y_{i,j} - y_{i,j}^*|}{y_{i,j}^*}
$$
### Squared Relative Difference
The squared relative difference is defined as:
$$
\text{srel} = \frac{1}{T} \sum_{i,j} \frac{|y_{i,j} - y_{i,j}^*|^2}{y_{i,j}^*}
$$
### RMS (Linear)
The root mean square error in linear space is:
$$
\text{RMS} = \sqrt{\frac{1}{T} \sum_{i,j} |y_{i,j} - y_{i,j}^*|^2}
$$
### RMS (Logarithmic)
The root mean square error in logarithmic space is:

$$
\log_{10} = \sqrt{\frac{1}{T} \sum_{i,j} |\log y_{i,j} - \log y_{i,j}^*|^2}
$$
## Ground Truth
Often obtained using LiDAR. Or multi sensor fusion (SLAM)


## Deeper Depth Prediction with Fully Convolutional Residual Networks 
Encoder Decoder CNN

### berHu Loss
$$ \mathcal{B}(x) = \begin{cases} |x|, & \text{if } |x| \leq c, \\ \frac{x^2 + c^2}{2c}, & \text{if } |x| > c. \end{cases}$$$$ \text{where } c = \frac{1}{5} \max_i(|\tilde{y}_i - y_i|)$$

## Deep Ordinal Regression Network for Monocular Depth Estimation

Intermediate between regression and classification.  

CNN Encoder with ASPP Architecture. 

#### Ordinal Regression
Discretize depth into ordinal levels. 
A logarithmic discretization makes the depth loss relative, not absolute. 

Fore each level output 2 feature maps with logits encoding that the pixel is at depth lower/higher than ordinal value. 

![[Pasted image 20250125165846.png]]

The **loss function** is defined as:

$$
\mathcal{L}(\chi, \Theta) = -\frac{1}{\mathcal{N}} \sum_{w=0}^{W-1} \sum_{h=0}^{H-1} \Psi(w, h, \chi, \Theta)
$$
where:
$$
\Psi(w, h, \chi, \Theta) = \sum_{k=0}^{l(w, h)-1} \log \left(P^k_{(w,h)}\right) + \sum_{k=l(w, h)}^{K-1} \log \left(1 - P^k_{(w,h)}\right)
$$
and:

$$
P^k_{(w,h)} = P\left(\hat{l}_{(w,h)} > k \mid \chi, \Theta\right) = \frac{e^{Y_{(w,h,2k+1)}}}{e^{Y_{(w,h,2k)}} + e^{Y_{(w,h,2k+1)}}}
$$

#### Inference
At inference, the predicted depth is computed as:

$$
\hat{d}_{(w,h)} = \frac{t_{\hat{l}(w,h)} + t_{\hat{l}(w,h)+1}}{2}
$$
where:

$$
\hat{l}_{(w,h)} = \sum_{k=0}^{K-1} \eta\left(P^k_{(w,h)} \geq 0.5\right)
$$
Essentially find the highest level with lower depth and output the average of this and the next higher level. 




## P3Depth: Monocular Depth Estimation with a Piecewise Planarity Prior
![[Pasted image 20250125175736.png]]
# Monocular Depth Completion
Given an input RGB image and a sparser depth map (from RGB-D camera, from LiDAR or Radar), the goal is to densify/complete the depth map. 

![[Pasted image 20250125180831.png]]
* Missing depth values are just filled in with zeros. 
* Fusion is done by channel concatenation

Example of **middle fusion**, first process the modalities (RGB, depth) separately and fuse features internally.  

# Stereo Depth Estimation


## Simplified Stereo Model
From the pinhole camera model: 
![[Pasted image 20250126101829.png]]
$$x^w = \frac{bu}{D}$$
$$y^w = \frac{bv}{D}$$
$$z^w = \frac{bf}{D}$$
where Disparity $D=u_l-u_r$ 

### Depth Resolution
$$\Delta Z = \frac{Z^2}{bf}*\Delta D$$

### Correspondences
To compute D, the pixel correspondences between the left and right image are necessary. 
This is traditionally done with SSD patch sliding. 

Challenging because of: 
* Occlusion
* Texture less regions
* Non-lamberitian surfaces


## Stereo vision – learning to compare image patches

### Siamese network 
Twin networks that use the same weights while working in tandem on two different inputs to minimize a distance metric for similar objects and maximize for distinct ones.

**Loss:**$$ \begin{cases} \min \|f(x^{(i)}) - f(x^{(j)})\|, & \text{if similar} \\ \max \|f(x^{(i)}) - f(x^{(j)})\|, & \text{otherwise} \end{cases} $$

## End-to-end learning of geometry and context for deep stereo regression
First pass each image through a almost siamese network (they share the weights), to produce 2D feature maps (3D tensor) for each image.

Then the Correlation layer comes up, which concatenates the 3D feature tensors in the channel dimension. The concatenation happens with one of the tensors being shifted by a disparity amount. This is done for no disparity up to a max disparity. Outputting multiple 3D tensors (one 4D tensor).
![[Pasted image 20250126104835.png]]

The following 3D convolutions keep the disparity channel the same size. 

In the output layer there is still a cost volume, but the feature channel was reduced to 1. 
Hence the output has dimensions $H * W * (D_{max}+1)$  
So for each pixel we have a "logits" vector for the possible disparities, but actually these are not logits, the rather are costs. Hence the are negated in the soft argmin. 

**Soft argmin**
$$\hat{d}[i, j] = \sum^{D_{max}}_{D=0}D*softmax(-C[i, j, D])$$
High cost means low probability. 

**Final loss:**  
$L_1$ regression loss on disparity values

$$
L = \frac{1}{HW} \sum_{i=1}^{H} \sum_{j=1}^{W} \left\| \hat{d}[i,j] - d[i,j] \right\|_1
$$


# Unsupervised Depth Estimation


## Unsupervised depth estimation – left-right consistency

On a **stereo** dataset (for supervision), train a model to predict a depth map from the left image.  Use the stereo model to construct the right image using the left image and the depth map. 

Take the loss between the reconstructed and the original right image. 

Uses a fully convolutional encoder-decoder network with skip connections (like seen in U-net)
![[Pasted image 20250126111806.png]]

Achieves real time performance. 

Results on **KITTI**
![[Pasted image 20250126112041.png]]


## Unsupervised Learning of Depth and Ego-Motion from Video

Replace the need for stereo supervision with motion. 

A CNN predicts the depth map of a frame and another CNN predicts the Pose between different frames. 

The pose and depth are used to do pixel wise warping of the current frame into the next frame, i.e. predicting the appearance of the next frame. 
The difference between the actual next frame and the prediction gives the loss signal. 


![[Pasted image 20250126120025.png]]

Results 
![[Pasted image 20250126120343.png]]

