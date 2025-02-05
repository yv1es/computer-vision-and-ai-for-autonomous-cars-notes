
# Visual grounding
Humans want to instruct the automated driving system (ADS)

Visual grounding is the task of given a **input scene** and a **verbal description** of an object in that scene, detect that referred object (i.e. ground the description to the scene)
It is also often called **referral-based object** detection. 

* **3D object localization**: detect the object directly in the point cloud
* **3D object identification:** classify the bounding box of the referred object among the set of bounding boxes of the same class in the scene

## Grounding-by-detection framework for referral-based 3D localization
Has a language backbone and a visual 3D backbone and **verbo-visual fusion**.
![[Pasted image 20250204183046.png]]

## Verbo-visual fusion in 3D visual grounding via transformers (3DVG-T)
![[Pasted image 20250204185630.png]]

## Multi-view transformer for 3D visual grounding (MVT-3DVG)
Spatial specifications are viewpoint dependent which poses a challenge for visual grounding.  

The idea of MVT-3DVG is, to generate multiple versions of the input 3D pcd (different **yaw** rotations).  
This is too expensive, hence **decouple** point cloud level feature extraction (invariant to affine transforms) from positional feature extraction for objects (view point dependent).  

![[Pasted image 20250205090513.png]]



## ConcreteNet
Learn the camera viewpoint. 

Uses **bottom-up attentive fusion** and adds an extra **camera token**. To supervise the camera token use it to predict the camera position. 

Only the camera position is predicted during training, since it normally oriented at the scene and has 0 roll. 

The attentive fusion again uses self and cross attention (queries from text, keys values from 3D). The attention is **masked** to make attention more local (attention scores for objects which are far apart are 0). This localness decreases with increasing layer depth.

![[Pasted image 20250205092103.png]]

Add contrast loss, which pulls together language embedding (anchor) and correct instance embedding and pushes away instance embedding and other object embeddings. 
![[Pasted image 20250205092441.png]]



# Anomaly segmentation
Predict a **soft dense outlier/anomaly** segmentation map in $[0, 1]$. 

Threshold the map with different values, calculate precision recall curve and obtain **AP** as the overall metric for anomaly segmentation. 

Another popular metric is the **false positive rate at 95% recall (FPR95)** useful for safety-critical applications. 

## Inference
Either use the thresholded map to generate a **hard anomaly mask** to combine with for example semantic segmentation.

Or keep the soft map for downstream modules (e.g. for planning)

## Training 
By definition, anomalies are not **out-of-distribution samples**, not in our training data. 

* **discriminative**:  train on in-distribution and condition (regularize) the model 
* **generative**: train on additional auxiliary negative data (often synthetic)  

## Bayesian DeepLab
Discriminative. 

Express the total uncertainty in the model’s predictions as a combination of:
- **Epistemic uncertainty**, i.e., uncertainty associated with the model
- **Aleatory uncertainty**, i.e., uncertainty associated with the data

**What we are after for anomaly segmentation is the epistemic uncertainty.**

- With only one sample and a deterministic network mapping, we cannot estimate the epistemic uncertainty.
- **Solution:** Monte Carlo integration using dropout. Multiple forward passes with the same input and different, randomly de-activated neurons.

Total uncertainty over $T$ Monte Carlo samples / forward passes: **predictive entropy of output**  
$$
\mathcal{H}(y \mid \mathbf{x}) = -\sum_{c} \left( \frac{1}{T} \sum_{t} y_c^t \right) \log \left( \frac{1}{T} \sum_{t} y_c^t \right)
$$

**Epistemic uncertainty**: mutual information between the output distribution and the weights distribution. A high value of epistemic uncertainty is an indication of an outlier.  
$$
\mathcal{I}(y, \mathbf{w} \mid \mathbf{x}) = \mathcal{H}(y \mid \mathbf{x}) - \frac{1}{T} \sum_{c, t} y_c^t \log y_c^t
$$
## Image re-synthesis
Generative. 

Components: 
1. A "normal" semantic segmentation model
2. The re-synthesize model (e.g. a conditional GAN) which takes as input the segmentation map and synthesizes a image. 
3. The discrepancy network, which is trained to detect inconsistencies between the original input image and the re-synthesized image. 

We take a image and its ground truth semseg labels and assign some instances a random label. Then the altered semseg labels are re-synthesized to an image. 
The **discrepancy network** is then trained to  detect the altered instances (given the re-synthesized image and the original image). 

![[Pasted image 20250205102252.png]]



