
Task of answering What? and Where? 
Becomes Instance segmentation when also requiring Segmentation Masks


# Metrics
$$Precision = \frac{TP}{TP + FP}$$
$$Recall = \frac{TP}{TP + FN}$$
$$IoU = \frac{\text{area of overlap}}{\text{area of union}}$$
False Positive $\Leftrightarrow$ IoU < threshold (or duplicate)   
False Negative $\Leftrightarrow$ mission detection 

## Average Precision (AP)

$AP[class, iou]$ average precision for specific class and IoU threhold. 
	1. Rank all predictions for $class$ according to their confidence
	2. Predictions are correct then IoU > $iou$ (thresholding)
	3. Build precision (y) - recall (x) curve
	4.  $AP[class, iou]$ is the area under the curve (AUC)

Often times before computing AUC, the curve is smoothed, by making it monotonically decreasing. 

For $AP[class]$ take the mean of $AP[class, iou]$  over a selection of $iou$. 
For overall $AP$ take the mean of $AP[class]$  over all classes. 
"AP is really an average, average, average precision"


# R-CNN

Uses non learning region proposal algorithm, producing a few thousand region proposals, also called region of interests (ROI). 

The ROI are cropped and warped to standard size and fed to a CNN doing classification and regression on the center coordinates and width and heights of the bounding box. 

Slow because each ROI has to be passed through the CNN. 

# Fast R-CNN
Compute feature map with a single forward pass through a large back bone CNN. 

Still use the same region proposal algorithm but now crop the features from the feature map and pass them to the box classification regression CNN. This CNN can be smaller since the backbone CNN did the feature extraction. 

The cropping of the feature map is not straight forward, as the feature map will not be of the same dimension as the input image. Use RoIPooling. 

## RoI Pooling
(first proposed in SPP-net) 
1. divide region of interest into grid of cells (size H x W). 
2. snap the RoI to integer coordinates of feature map (rounding)
3. do max pooling over each RoI cell, which yields a H x W, feature map.  


## Multi Task loss
Two sibling output layers:  

1. The first outputs a discrete probability distribution (per RoI), $p = (p_0, \ldots, p_K)$, over $K + 1$ categories. Cross-entropy loss.  

2. The second sibling layer outputs bounding-box regression offsets:  
   $$
   t^k = (t^k_x, t^k_y, t^k_w, t^k_h)
   $$
   for each of the $K$ object categories indexed by $k$.  

Each training RoI is labeled with a ground-truth class $u$ and a ground-truth bounding-box regression target $v$.  

$$
L(p, u, t^u, v) = L_{\text{cls}}(p, u) + \lambda [u \geq 1] L_{\text{loc}}(t^u, v)
$$
$u = 0$ means "no-object class".
$$ L_{\text{cls}}(p, u) = -\log p_u $$
(Loss for true class: cross-entropy) $$ L_{\text{loc}}(t^u, v) = \sum_{i \in \{x, y, w, h\}} \text{smooth}_{L_1}(t^u_i - v_i) $$
In which: $$ \text{smooth}_{L_1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases} $$
(This robust $L_1$ or Huber loss is less sensitive to outliers.)

# Faster R-CNN
Handcrafted RoI generators are still slow and might not perform well recall wise on some datasets. Hence Faster R-CNN gets rid of them using a region proposal network RPN. 

A large backbone CNN extracts image features. 
The RPN proposes RoI. 
RoI pooled feature maps are passed to the 2nd stage smaller CNN. 
## RPN
3 x 3 sliding window over the feature map. 
For every location and every anchor box, the RPN predicts an objectness score 
and does box regression (dx, dy, dh, dw), refining the anchor box. 

Anchor boxes are different scale and different aspect ration regions. 

### Loss and ground truth
The ground truth has to be computed beforehand, offline. 

Meaning that for each sample and anchor box, the anchor box is labeled as an object when its IoU is higher than 0.7. And it is labeled a empty when IoU is smaller than 0.3. 

Boxes in between are not considered in training (do not influence the loss). 

Notice that one ground truth bounding box might have multiple anchor boxes associated with it. 

Finally use loss: 
$$
\mathcal{L}(\{p_i\}, \{t_i\}) = \frac{1}{N_{\text{cls}}} \sum_i \mathcal{L}_{\text{cls}}(p_i, p_i^*) + \lambda \frac{1}{N_{\text{reg}}} \sum_i p_i^* \mathcal{L}_{\text{reg}}(t_i, t_i^*)
$$
For all negative and positive samples. ($N_{reg} = \#positives$)  


## Training 
First optimize RPN alone. Then fix the RPN and train 2nd stage. 
Finally unfreeze and connect stages and train them jointly. 


# Faster R-CNN with Feature Pyramid Network (FPN)

Construct pyramid of different scale convolutional features. 

Improving scale invariance. 

![[Pasted image 20250126150425.png]]



# Direct set prediction for object detection
Issues with indirect detectors
* Heuristic IoU parameter for ground truth generations
* Multiple anchors for single ground truth box, requiring non-maxima suppression $\Longrightarrow$ duplicate predictions (**expensive non-max suppression** is required)

## Loss formulation
![[Pasted image 20250201154124.png]]

Define $\mathcal{L}_{match}$ as the matching cost between ground truth and predicted bounding box. 
Use this to find an minimal cost matching using the Hungarian algorithm. 

![[Pasted image 20250201155628.png]]

Then use the found matching to define the **final training loss $\mathcal{L}_{Hungarian}$** 
![[Pasted image 20250201155823.png]]
Notice that for the matching cost the prediction probability is not log-ed, to keep its value in the same range as the $\mathcal{L}_{box}$ regression loss. Then for the final training loss the log is added. 


## Detection Transformer (DETR)

![[Pasted image 20250201162211.png]]

![[Pasted image 20250201162308.png]]