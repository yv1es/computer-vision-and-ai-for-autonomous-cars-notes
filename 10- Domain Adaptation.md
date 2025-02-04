# Domain shift
Domain shift leads to poor performance. 
![[Pasted image 20250203102436.png]]

## Types of domain shift
Low level
* Condition-level shifts (e.g. day-night shift)
* Difference in sensor

High level
* semantic shift in size, shape, spatial configurations (e.g. different locations, bigger vehicles in US) 

Rendering: synthetic vs real, can comprise both levels

## Formally 
![[Pasted image 20250203103317.png]]

## Naïve approach
Represent "outsiders" in the training set. 
* Curse of dimensionality (combination of different domain shifts)
* Unforseen data at test time
* Data acquisition is hard

# Domain adaptaion to address domain shift

## Training scenarios
**Unsupervised domain adaptation**
Source domain: labeled data 
Target domain: unlabeled data

**Weakly supervised domain adaptation**
Source domain: labeled data
Target domain: unlabeled data with correspondences to source domain

**Source-free domain adaptation or model adaptation**
Source domain: pretrained model 
Target domain: unlabeled data

**Test-time or online domain adaptation**
Source domain: pretrained model
Target domain: unlabeled sequential test data

**Domain generalization**
Source domain(s): labeled data 
Target domain(s): unseen during training, no adaptation at test time

## Avenues for domain adaptation
1. **Input-level adaptation**: align the image distributions of the source and the target domain. Suitable for **low-level domain shifts**.
![[Pasted image 20250203105123.png]]

2. **Feature-level adaptation**: align the feature distributions of the source and the target domain.
![[Pasted image 20250203105209.png]]

1. **Output-level adaptation**: align the outputs for the source and the target domain
![[Pasted image 20250203105249.png]]


## Input-level adaptation
### Physics-based domain translation
no target data needed; reliable, controllable and interpretable but **not always possible**

#### Fog simulation on clear-weather scenes
![[Pasted image 20250203110312.png]]
$\textbf{L}$ is the "atmospheric" light, an RGB color vector close to 1. 
When the transmittance $\hat{t}$ is large, we take the clear weather color, when it is small, we take more of $\textbf{L}$ 
When a point is far from the sensor or when $\alpha$ is high, then $\hat{t}$ is small. 

**Foggy Cityscapes** is an example where the fog simulation is applied to Cityscapes. 
The original annotations are used. 

#### Fog simulation on clear-weather scenes using semantics
![[Pasted image 20250203111832.png]]
Filter transmittance map using the semantic segmentation labels. 
![[Pasted image 20250203111934.png]]

#### Rain simulation on clear-weather scenes
Uses the fog simulation with transmittance 
![[Pasted image 20250203113438.png]]

And an additional particle simulation, to compute "rain streaks" 
![[Pasted image 20250203113632.png]]
![[Pasted image 20250203113703.png]]

#### Lidar snowfall simulation 
![[Pasted image 20250203150513.png]]
![[Pasted image 20250203150525.png]]
![[Pasted image 20250203150545.png]]

### Learned domain translation
target data needed; can learn complex domain shifts involving multiple factors of change; **not very reliable, controllable or interpretable**

![[Pasted image 20250203150754.png]]

Can be learned from **paired or unpaired data**. 
Is easier with paired data, but paired data means perfect pixel alignment, which is not the case for autonomous driving datasets. There is much more unpaired data.![[Pasted image 20250203151251.png]] 

#### Generative adversarial networks – GANs

![[Pasted image 20250203151508.png]]

Basic conditional GANs suffer from **mode collapse**. 

### CycleGAN for unpaired image translation

Train two translators, $F$ from domain $A$ to $B$ and $G$ from $B$ to $A$ and add a cycle-consistency loss to the optimization.  $||G(F(x)) - x||$  
![[Pasted image 20250203152023.png]]

##### Results
![[Pasted image 20250203152046.png]]


### Hand-crafted domain transformation
e.g. in frequency space: target reference image needed; controllable and interpretable; limited in representation to the attributes of one image; not always reliable

#### FDA
Replace the low frequency amplitudes in Fourier domain. 

Parametrized by $\beta$ where higher $\beta$ means more domain shift but also more artifacts. 
This allows to train diverse models with different $\beta$ values and the "ensemble" these models.

![[Pasted image 20250203152556.png]]

## Feature-level adaptation

### CyCADA: Cycle-consistent adversarial domain adaptation
Based on CycleGAN. 

Adds a **feature level discriminator** to make the features from images of the source domain indistinguishable from the features of images of the target domain.  

Also adds a **Semantic consistency loss** which makes the predictions of the original source model for a source image equal to the prediction on the translated image. 

![[Pasted image 20250203153742.png]]

### Condition-invariant semantic segmentation (CISS) 
Uses hand crafted domain translation and a siamese network. 
Optimizes the encoder to be invariant to domain shifts. 
![[Pasted image 20250203161158.png]]

### Contrastive model adaptation (CMA)
Source free, meaning that there is no labeled ground truth but we have a model trained on the source domain. And image pairs of source and target domain (weak supervision). 

Divide source and target image into patches and warp them so that they align.
Take a patch in the source domain, it is the anchor. 
Then use the aligned patch in the target domain as a positive and the other patches from the source domain in the same image as negatives. 

In feature space, this pulls together features from the same patch from different domains and push apart different patches from the same domain.

![[Pasted image 20250203163006.png]]

The patches are aligned using Refign.
A patch feature is the average of the pixel features. 
Only patches with average confidence above 0.2 are considered. 
![[Pasted image 20250203163129.png]]

The source model is a Encoder Decoder segmentation model (e.g. could be a U-net) but during domain adaptation with the $\mathcal{L}_{cdc}$ loss, the decoder is frozen, because we want to optimize the features. 

![[Pasted image 20250203163542.png]]
Finally there is also a projection head after the encoder which is only there to allow some flexibility among the features.

#### Results
![[Pasted image 20250203163816.png]]

We see that the features are no longer clustered by domain but mostly by class. 

## Output level adaptation

### Adversarial discriminative domain adaptation

#### AdaptSegNet
Make the output from target domain samples look similar to the output from source domain samples. 

Method similar to adversarial feature-level adaptation in CyCADA
But with 
* Adversarial learning directly on the output space 
* Pixel level discriminator for segmentation outputs (opposed to discriminating a complete network output, e.g. image)

![[Pasted image 20250203171702.png]]

### Self-training (ST) setting for domain adaptation
1. In the supervised case we have labels $\mathbf{y_s}$ for the source domain and $\mathbf{y_t}$ for the target domain. 
![[Pasted image 20250203172839.png]]

2. In the unsupervised case - **default self-training** -  there are only pseudo labels $\mathbf{\hat{y}_t}$ for the target domain. 
![[Pasted image 20250203172852.png]]

1. Unsupervised case - **curriculum self-training**
	* alternate between optimizing pseudo labels and network weights
![[Pasted image 20250203173102.png]]

##### Extension: class-balanced self-training
![[Pasted image 20250203173237.png]]
##### Curriculum
![[Pasted image 20250203173414.png]]

### Domain adaptation via cross-domain mixed sampling (DACS)
Due to potentially large domain gap, the network may learn to discern between the domains. 

**Idea**:  create artificial mixed images and labels with content from the source and target domain. 
* The labels for these new images are partly ground-truth, so they are not conflated for entire images
* Properly labeled and pseudo-labeled pixels can be neighbors, making the implicit discerning between domains unlikely, since it would have to be done at a pixel level.

![[Pasted image 20250203182137.png]]

## Cross-domain correspondences in weakly supervised domain adaptation

Improve the pseudo labels by predicting labels on the source images and matching the target image (based on GPS for example) and then **fix alignment** and dynamic obstacles. 

### Probabilistic learned warping for output alignment

We have a image pair $\mathbf{I}, \mathbf{J}$ and we want to predict the warp (dense 2D displacement field $\mathbf{F_{J\rightarrow I}}$) 

1. Use a synthetic known warp $\mathbf{W}^{-1}$ to generate $\mathbf{I'}$ from $\mathbf{I}$. 
2. Let the network predict $\mathbf{F_{I'\rightarrow I}}$  which should be $\mathbf{W}$
3. Let the network predict $\mathbf{F_{J\rightarrow I}}$ and $\mathbf{F_{I\rightarrow J}}$  then the composition  $\mathbf{F_{I' \rightarrow J \rightarrow I}}$  should be $\mathbf{W}$
   
![[Pasted image 20250204090637.png]]


Do model the warps as Gaussians to model uncertainty. 

![[Pasted image 20250204091907.png]]
![[Pasted image 20250204091948.png]]

## Test-time domain adaptation
### Test entropy minimization (TENT) 
No training data at hand, just the test data. Need to adapt the network on the fly. 
Do a parameter update after every batch. 

But the neural network is too high dimensional and non-linear for it to be optimized by a single pass. As a solution **only optimize the (batch) normalization parameters**, as they are linear and process individual channels. 

**Recall**: in BatchNorm, after training, the statistics of the activations is computed across the entire training set to compute the final parameters, which remain fixed for inference. 

For TENT discard the normalization statistics from the train set, and compute them adaptively for each test batch. Compute the scaling and shifting parameters via back propagation.  


