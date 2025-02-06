# Instance segmentation
Instance segmentation is object detection with pixel-level masks instead of bounding boxes for objects. (Not segmenting uncountable objects, e.g. sky, grass)

## Metrics 
### AP 
Same as in [[6 - Object Detection#Metrics#Average Precision (AP)]] but IoU is computed at the pixel level.

## Proposal based
### Mask R-CNN
Based on [[6 - Object Detection#Faster R-CNN]]. 

Detect then segment method. 
![[Pasted image 20250130105249.png]]

Replaces [[6 - Object Detection#RoI Pooling]] with RoI Align. 

#### RoI Align
Transform arbitrary-sized proposal into a fixed-dimensional representation (e.g., 2x2 in this toy example). In practice, 7x7 or 14x14 feature maps after RoIAlign in Mask R-CNN.

![[Pasted image 20250130105921.png]]

For mask inference the predicted mask (14 x 14 or 28 x 28) is resized to the RoI size and thresholded at 0.5. Due to the upsampling, the masks are normally not very sharp. 

#### Mask R-CNN loss
The multi-task loss is defined as: $$ L(p, u, t^u, v) = L_\text{cls}(p, u) + [u \geq 1] L_\text{box}(t^u, v) $$
For each proposal, we have a $K \times m \times m$ feature map $f$. This is passed through a sigmoid layer to get sigmoid scores $s$. 

The Mask R-CNN loss for instance segmentation includes a **binary segmentation cross-entropy loss** $L_\text{mask}$ on the sigmoid scores for the mask that corresponds to the ground-truth class $u$ with mask $w$. 

The full loss is: $$ L(p, u, t^u, v, s^u, w) = L_\text{cls}(p, u) + [u \geq 1] L_\text{box}(t^u, v) + [u \geq 1] L_\text{mask}(s^u, w) $$ 
The binary segmentation loss $L_\text{mask}$ is defined as: 
$$ L_\text{mask}(s^u, w) = -\frac{1}{m^2} \sum_{i=1}^{m^2} \Big( (1 - w[i]) \log(1 - s^u[i]) + w[i] \log(s^u[i]) \Big) $$

## Proposal free 
Downside of proposal-based methods: 
* low-resolution mask prediction
* not very fast inference
* even more time at inference for further processing (e.g. non-max-suppression)
* near duplicate predictions

Idea: View instance segmentation as a direct dense prediction task, no detect then segment. 
* gives high resolution mask (as [[4 - Semantic Segmentation]])
* fast inference, normal forward pass


### Spatial embedding 

Regress the offset of each pixel to its ground-truth instance centroid. 
$$ \mathcal{X} = \{x_1, \ldots, x_N\} $$
$$ \mathcal{S} = \{S_1, \ldots, S_Z\} $$
A **pixel** \(x_i\) is assigned to an instance \(S_k\) by learning to **point to its instance's spatial center**: 
$$ C_k = \frac{1}{|S_k|} \sum_{x \in S_k} x $$


There is a need to assign pixels to centroids uniquely. 
Just clustering by centroids does not allow for gradient-based optimization. 

### Joint optimization of spatial embeddings and clustering bandwidth


![[Pasted image 20250130134943.png]]

![[Pasted image 20250130135244.png]]
![[Pasted image 20250130135329.png]]


Foreground/background probability map $\phi_k$ for instance $S_k$ is directly supervised via the **Lovasz-hinge loss**, which is a convex surrogate to the **Jaccard loss** (Jaccard index = Intersection over Union), with respect to the ground-truth hard instance mask.

No need to account for class imbalance between foreground and background, from which the standard cross-entropy loss would suffer.

The spatial embeddings $o_i$ and $\sigma_i$ are supervised implicitly via backpropagation from the loss computed for $\phi_k$.
![[Pasted image 20250130140424.png]]

# Panoptic segmentation
![[Pasted image 20250130140556.png]]

**Dataset** (training set) for generic supervised learning setup:

$$
\mathcal{D} = \{(x_i, y_i) : i = 1, \ldots, N\}
$$

**Pairs of inputs and reference outputs**

For panoptic segmentation:

- **Input:** 
  $$
  x_i = \hat{I}_i \in \mathbb{R}^{H \times W \times 3}
  $$
  image (RGB)

- **Output:** 
  $$
  Y_i \in \{1, \ldots, C\} \times \{1, \ldots, Z\}^{H \times W}
  $$
  encoding of the **class labels** and the **instance labels**, where \(Z\) is the maximum number of instances for a given class in one image (typically set to a very high but finite value).

## Evaluation: Panoptic Quality (PQ) metric


First step is segment matching (prediction and ground truth). Match segments with IoU greater than 0.5. Guarantees unique matching. One ground truth segment for uncountable, can be disconnected. 

![[Pasted image 20250130154608.png]]


## Panoptic FPN

![[Pasted image 20250130161420.png]]

Do instance segmentation and semantic segmentation and merge the outputs. 

![[Pasted image 20250130161517.png]]


## MaskFormer
Set of ground-truth segments:  
$$z^{gt} = \{(c_i^{gt}, m_i^{gt}) | c_i^{gt} \in \{1, \dots, K\}, m_i^{gt} \in \{0, 1\}^{H \times W}\}_{i=1}^{N^{gt}}$$
Output as set of probability-mask pairs:  
$$z = \{(p_i, m_i)\}_{i=1}^N \quad \text{(soft mask values in } [0, 1])$$
Find the optimal matching between the ground-truth set and the output set with the Hungarian algorithm for bipartite matching, as in object detection.

Pad ground-truth set with "no object" tokens to allow one-to-one matching.

Need to define the **matching cost** for each (predicted mask, ground-truth mask) pair.

Matching cost for each (predicted mask, ground-truth mask) pair:  
$$\mathcal{L}_{\text{match}}(z_{\sigma(i)}, z_i^{gt}) = -[c_i^{gt} \neq \emptyset] p_{\sigma(i)}(c_i^{gt}) + [c_i^{gt} \neq \emptyset] \mathcal{L}_{\text{mask}}(m_{\sigma(i)}, m_i^{gt})$$  
$$\mathcal{L}_{\text{mask}}: \text{binary mask overlap loss}$$  
Avoid logarithm in probabilities for the classification term of matching cost, in order to make this term commensurate with the mask overlap term.

Set loss (Hungarian loss) given the optimal match $\hat{\sigma}$ found by the Hungarian algorithm: **mask-classification loss**  
$$\mathcal{L}_{\text{mask-cls}}(z, z^{gt}) = \sum_{i=1}^N -\log(p_{\hat{\sigma}(i)}(c_i^{gt})) + [c_i^{gt} \neq \emptyset] \mathcal{L}_{\text{mask}}(m_{\hat{\sigma}(i)}, m_i^{gt})$$

![[Pasted image 20250206100710.png]]
* No transformer encoder (unlike DETR) 
* Pixel decoder for per pixel features (unlike DETER). The masks are predicted as the dot product between mask and pixel embeddings (apply sigmoid). 
* can be trained on semantic-, instance- and panoptic segmentation

Different **inference processes** are applied in practice.
- **Panoptic and instance segmentation**: for each pixel, predict the dominant probability-mask pair  $$\arg \max_{i: c_i \neq \emptyset} p_i(c_i) \cdot m_i[h, w]$$  both the probability of the most likely class and the mask probability should be high.
- **Semantic segmentation**: marginalize over all probability-mask pairs to predict the semantic class   $$\arg \max_{c \in \{1, \dots, K\}} \sum_{i=1}^N p_i(c) \cdot m_i[h, w]$$
MaskFormer is not efficent to train and less performant than specialized architectures. 


## Mask2Former
Improvements: 
1. mask the image feature cross attention
2. do cross-attention before self attention in decoder block 
3. pass higher resolution image features to transformer decoder
4. $\mathcal{L}_{mask}$ in the matching loss is computed for every ground truth, prediction pair ($O(n^2)$. Use a approximation: 
	1. Uniform sampling of points for the matching cost
	2. importance sampling for the final mask loss

But still needs to be re-trained for different tasks. 

## OneFormer
One model for semantic- instance and panoptic seg. 

Generate a task text (e.g. "a photo with a car, a photo with a car, ...")
![[Pasted image 20250206111806.png]]
The text part is only there for training. At inference the queries are just initialized with the task token. 
