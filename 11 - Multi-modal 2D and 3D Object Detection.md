
# Sensor fusion taxonomy

* **Late fusion**: fuse individual results from the sensors
![[Pasted image 20250204095549.png]]

* **Early (low-level) fusion**: fuse raw sensor data
	* for lidar-camera fusion: also when combining raw lidar data with image features
![[Pasted image 20250204095645.png]]

* **Mid-level/deep fusion:** fuse internal features from the two sensors
![[Pasted image 20250204095741.png]]

* **Asymmetric fusion:** fuse object-level representations from one sensor with **feature-level** or data-level representations from the other
![[Pasted image 20250204095941.png]]

* **Asymmetric fusion**: fuse object-level representations from one sensor with feature-level or **data-level** representations from the other  ![[Pasted image 20250204100042.png]]

* **Weak fusion**: use one sensor solely for providing guidance to the other for generating features and/or results
![[Pasted image 20250204100153.png]]


## Late vs. early vs. deep fusion

### Early fusion 
inherent discrepancies (alignment, representation, sparsity) 

### Deep fusion
addresses early fusion deficits but is complex and sensitive to sensor calibrations and requires multi-modal data

### Late fusion
Can use unimodal data for specific detectors and use little multi modal data for training the fusion. 
Fusion amounts to pruning false positives from the union of the detectors. Hence unimodal detectors should be optimized for recall. 


# Lidar-camera fusion for 3D object detection
Improve lidar-based 3D detection networks by fusing information from the camera

Cameras provide high-resolution features with rich color texture which can
* increase the recall on distant objects
* improve the localization accuracy

## Frustum PointNets: weak lidar-camera fusion
Use a 2D object detector and and only search for the object in 3D within the frustum of the 2D detection. 

![[Pasted image 20250204101722.png]]


## Multi-view 3D object detection network (MV3D)
Project 3D lidar point cloud to 2D birds eye view. 
![[Pasted image 20250204105223.png]] 
The bird’s eye view map has **advantages** over the front view and camera view
* preserves physical sizes, thus having smaller size variance.
* avoid the occlusion problem
* In road scenes, BEV is often sufficient since objects typically lie/stand on the ground plane

But: 
* Coarse (sparse data), information might get lost especially for small objects
* Cannot handle multiple objects along the vertical dimension

## PointPainting: early lidar-camera fusion
Poor geometric cues in lidar point clouds to discern distant objects’ semantics

Example: pedestrian and pole appear similar to lidar
![[Pasted image 20250204105924.png]]
Use a 2D semantic segmentation model and project the softmax scores of the pixels onto the points in the point cloud. (append the softmax vector)


## Multimodal VoxelNet (MVX-Net): early lidar-camera fusion variant
Same idea as point painting, but not the segmentation scores are used, but the features from a VGG detection backend. Still a per point augmentation. 

## MVX-Net: mid-level lidar-camera fusion variant
Here the fusion is done at voxel level. Projecting the voxels onto the feature map and doing RoI pooling. Only use non empty voxels. 

The fusion is deeper in the network, making it more memory efficient. 


## PointAugmenting: early and mid-level lidar-camera fusion

Semseg scores are not that relevant for box localization in 3D. 
The MVX-Net VGG features are more relevant, but they are rather high-level and global. 

Idea: Use mid-level 2D detection features, which preserve more find detail. 
![[Pasted image 20250204111342.png]]

### Limitations of point-level fusion
* simple addition and concatenation might deteriorate the performance when image features are not good (e.g. low light)
* the hard association between one point and one image feature ignores the neighboring feature
* very sensitive to extrinsics calibration of lidar and camera


## TransFusion: mid-level lidar-camera fusion with soft association
Soft association of mid level features via cross-attention. 

Predict a class-specific heatmap in BEV, non-mamixa-supress and select top-N candidates as **queries** (position and features). 

The keys and values come from collapsing the image features along the vertical axis (max pool over columns). This is more efficient under the assumption that there is only one object per image column. 
![[Pasted image 20250204112928.png]]

### Spatially modulated cross attention (SMCA) 
Softly constrain attention to a 2D region around the predicted object center
1. Project query position from BEV space to the 2D camera view: $(c_x, c_y)$
2. Multiply cross-attention map element-wise with a soft Gaussian mask $$M_{ij} = \exp\left(-\frac{(i - c_x)^2 + (j - c_y)^2}{\sigma r^2}\right)
$$
## Aggregate view object detection (AVOD) 

### Mid level fusion
Operates on bird eye view point cloud. 
![[Pasted image 20250204120607.png]]

### Late fusion
![[Pasted image 20250204121108.png]]


## Camera-lidar object candidates (CLOCs) 
Does late fusion. 

Scores initial 3D box predictions with the help of 2D predictions. 

Project 3D boxes onto 2D image, a when projected box overlaps with 2D detection box, create a feature vector containing the IoU, confidences and distance between centers. 

![[Pasted image 20250204134107.png]]

# Radar-camera 3D fusion
The radar does not provide very accurate 3D data, we can not use the radar as the primary sensor for 3D detection.  Hence here we try to improve the 3D camera detections using radar. 
But radar gives accurate depth and speed estimation. 

## CenterFusion: asymmetric radar-camera fusion
Backproject 2D pointcloud to 3D frustum and convert BEV radar points into pillars. 
When pillars intersect the frustum take the closest one and use its features (x, y, depth, speed) and add them to the image features. 

![[Pasted image 20250204135327.png]]



# Multi-modal 2D object detection
![[Pasted image 20250204135718.png]]

Complementary sensors: 
**Lidar**: does not need illumination, richt 3D structure but sensitive to paritcles (bad weather)
**Radar**: does not need illumination, only coarse 2D and noisy but robust to weather  

## CRF-Net: camera-radar fusion for 2D detection
Convert radar detection into 3 meter high vertical pillars, project the pillars into the image and paint them (add channels) with radar distance and cross section, to form a **radar image**.
![[Pasted image 20250204140959.png]]

### Multi level fusion
Concatenate the radar channels at multiple layers to let the network learn at which depth to fuse. 
![[Pasted image 20250204141102.png]]

The network is pretrained on image data alone. Then use **BlackIn**, deactivate image input neurons with fixed probability (0.2) to make the model depend more on the radar data.


# Robust sensor fusion (in 2D)
So far we assumed consistent and redundant sensor streams but in practice we have asymmetric failures, due to adverse weather for example. 

We need to fuse adaptively, ignoring noise and degradations.  

## Entropy based

### Entropy estimation in 2D 
- Local, patch-level **measurement entropy**
- Assumption: the higher the measurement entropy, the higher the SNR
- All modalities are projected to the camera view for 2D detection
- For each channel $I$ of a sensor:
$$
p_i^{mn} = \frac{1}{MN} \sum_{j=1}^M \sum_{k=1}^N \delta(I(m+j, n+k) - i)
$$
where $M, N$ are patch dimensions.
- Estimate of local measurement entropy:
$$
\rho^{mn} = \sum_{i=0}^{255} p_i^{mn} \log(p_i^{mn})
$$

### Deep entropy-based adaptive fusion
Deep convolutional architecture, where the different sensor branches consist of blocks, where the feature maps are concatenated and weighted by their entropy. 
![[Pasted image 20250204143207.png]]

### Limitations 
Assumes higher entropy implies higher SNR, which might not be true (e.g. lidar and snow). 
Rather, the SNR should be learnable.
Also the deep entropy based method does scale quadratically with the number of modalities in terms of memory and computation. 

## HRFuser: multi-resolution sensor fusion for 2D detection with several modalities
Threats camera s primary modality. 
Generalizes [[4 - Semantic Segmentation#HRNet]] 

Creates different resolutions for the camera.
![[Pasted image 20250204154257.png]]

### Multi window cross attention (MWCA) fusion block
![[Pasted image 20250204154113.png]]

### Results 
![[Pasted image 20250204154423.png]]
