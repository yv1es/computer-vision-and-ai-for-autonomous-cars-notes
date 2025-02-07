# Semantic Segmentation 
Assign each pixel in the input image individually to a single class $C$ from set $\mathcal{C} = \{1,...,C\}$  


## Categorical Cross-Entropy Loss on Softmax Outputs

### Per Pixel Loss:
$$
-\log\left(\hat{Y}[h, w, y[h, w]]\right) = -\sum_{c=1}^{C} Y[h, w, c] \log\left(\hat{Y}[h, w, c]\right)
$$

### Categorical Cross-Entropy Loss for the Entire Image:
$$
\mathcal{L}_{CE}(\hat{Y}; Y) = -\frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} \sum_{c=1}^{C} Y[h, w, c] \log\left(\hat{Y}[h, w, c]\right)
$$

## Evaluation metrics
* **mIoU** : mean Intersection over Union (average over classes)
* **Pixel accuracy**: the percentage of pixels in the image which were correctly classified


## In-network Upsampling
### Unpooling
#### Nearest Neighbor
Just duplicate the value e.g. $2 \rightarrow \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix}$  for each pixel in the feature map. 
#### Max Unpooling
Keep information on which element was max, called switches. 
Take the value from switch position and pad with zeros. 
e.g. $2 \rightarrow \begin{bmatrix} 0 & 2 \\ 0 & 0 \end{bmatrix}$ 
#### Transpose Convolution (Deconvolution)
Output contains copies of the filter weighted by the input.
Sum at areas of overlap.
May need to crop the output to make it exactly 2 times as large as the input.

## Examples
### U-Net
Sequence of deconvolutions and concatenations with high-resolution features from the contracting path. 

### Atrous convolution (aka Dilated Convolution)
![[Pasted image 20250125153324.png]]
![[Pasted image 20250125153641.png]]

### Transformer
Tokenization becomes a problem as large tokes lead to imprecise segmentation while small tokens are too expensive. A solution is **shifted window attention (Swin)**.
![[Pasted image 20250125154717.png]]
Swin transformer can be used as a drop-in replacement of CNN backbones for the
encoder part of standard semantic segmentation networks.

### HRNet
There are three fundamental differences of HRNet from low-resolution classification networks and high-resolution representation learning networks:
1. High- and low-resolution convolutional layers are connected in parallel rather than in series.
2. High resolution is maintained throughout the entire network instead of recovering high resolution from low resolution.
3. Multi-resolution representations are fused repeatedly, rendering rich high-resolution representations with strong position sensitivity.

Downsampling uses strided convolution. 
![[Pasted image 20250125155241.png]]


On **Cityscapes** ![[Pasted image 20250125155356.png]]

