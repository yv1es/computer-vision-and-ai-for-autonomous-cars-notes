

###### Etc
**Compute mean** $\mu_j=\frac{1}{N}\sum_{i=1}^N x_{ij}$  
**Compute stdev** $\sigma_j=\sqrt{\frac{1}{N-1}\sum_{i=1}^N\bigl(x_{ij}-\mu_j\bigr)^2}$  
**Sigmoid** $\sigma(x) = \frac{1}{1 + exp(-x)}$  
**Softmax** $softmax(z)_i = \frac{exp(z_i)}{\Sigma_jexp(z_j)}$ 
**Cross Entropy Loss:** $-\Sigma_{x \in classes} y(x)log(p(x))$ 
**SGD with momentum** ($\alpha$  slightly below 1): 
$\mathbf{v} \leftarrow \alpha \mathbf{v} - (1 - \alpha) \eta \nabla_\theta \left( \frac{1}{B} \sum_{i=1}^B \mathcal{L}(f(x_i; \theta), y_i) \right)$ with $\theta \leftarrow \theta + \mathbf{v}$
**Kaiming initialzation:** 0-mean Gaussian with stddev = $\sqrt{2/d_{l-1}}$ 
**Batch Norm. (BN):** First each unit is normalized (subtract mean and divide by variance of the mini-batch). Then secondly scale (by $\gamma$) and shift (by $\beta$) (learnable $\gamma, \beta$).  
**Conv** $y[p,q]=\sum_{i=1}^m\sum_{j=1}^n w[i,j]\,x[p+i-1,q+j-1],\;p=1\ldots H-m+1,\;q=1\ldots W-n+1$  
**T. Conv** $y[p,q]=\sum_{i\in I}\sum_{j\in J}x[i,j]\,w[p-i+1,q-j+1],\;p=1\ldots H+m-1,\;q=1\ldots W+n-1$  
**Conv. output dimension:** $\lfloor \frac{H + 2P - k}{s} \rfloor + 1$   
**Attention:** $X_l = \text{softmax}(Q_l K_l^\top) V_l + X_{l-1}$   
**Masked:** $X_l = \text{softmax}(\mathcal{M}_{l-1} + Q_l K_l^\top) V_l + X_{l-1}$  w. $\mathcal{M}_{l-1}(x, y) = \begin{cases} 0 & \text{if } \mathbf{M}_{l-1}(x, y) = 1 \\ -\infty & \text{otherwise} \end{cases}$
**berHu Loss:** $B(x)=\begin{cases}|x|,&|x|\le c\\ \tfrac{x^2+c^2}{2c},&|x|>c\end{cases}$ with $c=\tfrac{1}{5}\max_i|\tilde{y}_i - y_i|$.  
**Huber Loss:** $= 0.5x^2 \text{ if } |x| < 1, \, |x| - 0.5 \text{ otherwise}$
**Polar**: $\begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} r \cos \alpha \cos \epsilon \\ r \sin \alpha \cos \epsilon \\ r \sin \epsilon \end{bmatrix}$ elevation $\epsilon$  azimuth $\alpha$ 
**Simplified stereo model** $x^w = \frac{bu}{D}$   $y^w = \frac{bv}{D}$   $z^w = \frac{bf}{D}$  where Disparity $D=u_l-u_r$ 
**Depth resolution**: $\Delta Z = \frac{Z^2}{bf}*\Delta D$
**Precision** $= \frac{\text{TP}}{\text{TP} + \text{FP}}$      **Recall** $= \frac{\text{TP}}{\text{TP} + \text{FN}}$
##### Self Driving Levels
**1** Feet off   **2** Hands off  **3** Eyes off  **4** Mind off  **5** Mind off, No ODD restriction 
###### LiDAR
**Sinusoidal Pulse** $P_T(t)=\begin{cases}P_0\,\sin^2\!\bigl(\frac{\pi t}{2\,\tau_H}\bigr),&0\le t\le2\,\tau_H\\0,&\text{otherwise}\end{cases}$  
**Square-Exponential Pulse** $P_T(t)=\begin{cases}C\,P_0\,\bigl(\frac{t}{\tfrac{\tau_H}{1.75}}\bigr)^2\,\exp\!\bigl(-\tfrac{t}{\tfrac{\tau_H}{1.75}}\bigr),&t\ge0\\0,&t<0\end{cases}$  
**Impulse Response** $H_T(R)=\rho_0\,\delta\bigl(R-R_0\bigr)$ clear weather 
**Received Power** $P_R(R)=C_A\,\frac{P_0\,\rho_0}{R_0^2}\,\sin^2\!\bigl(\frac{\pi\,(R-R_0)}{c\,\tau_H}\bigr),\,R_0\le R\le R_0+c\,\tau_H;\,0,\text{otherwise.}$  
**Symbols** $\tau_H:$ half-power width, $P_0:$ peak power, $c:$ speed of light, $\rho_0:$ diff. refl., $R_0:$ range, $C_A:$ const, $C:$ scale.
###### Radar
**Radar equation** $P_\text{received}=\frac{P_\text{transmitted}\,G\,\sigma\,A_e}{16\,\pi^2\,r^4\,L}$  
$G:$ gain, $\sigma:$ radar cross section, $A_e:$ aperture, $L:$ loss factor
##### Depth prediction
**Make3D** superpixels as planar via MRF plane coeffs.  
###### Metrics
**Threshold**: % of pixels s.t. $\max(\tfrac{y_i}{y_i^*},\tfrac{y_i^*}{y_i})=\sigma<\text{thr}.$  (common $thr.$ is $1.25^i$)
**Abs rel diff**: $\text{rel}=\tfrac{1}{T}\sum_i \tfrac{|\,y_i - y_i^*\,|}{y_i^*}.$   **Sq rel diff**: $\text{srel}=\tfrac{1}{T}\sum_i \tfrac{|\,y_i - y_i^*\,|^2}{y_i^*}.$  
**RMS**: $\sqrt{\tfrac{1}{T}\sum_i (\,y_i-y_i^*\,)^2}.$   **RMS(log)**: $\sqrt{\tfrac{1}{T}\sum_i|\log(y_i)-\log(y_i^*)|^2}.$  
###### Ordinal Regression
**Ordinal Loss**  $L(x,\Theta)=-\tfrac{1}{N}\sum_{w=0}^{W-1}\sum_{h=0}^{H-1}\Bigl[\sum_{k=0}^{l(w,h)-1}\log(\mathcal{P}^{k}_{(w, h)})+\sum_{k=l(w,h)}^{K-1}\log\bigl(1-\mathcal{P}^{k}_{(w, h)}\bigr)\Bigr].$
**Inference**  $\hat{l}(w,h)=\sum_{k=0}^{K-1}\eta\bigl(\mathcal{P}^{k}_{(w, h)}\ge0.5\bigr),\;\hat{d}(w,h)=t_{\hat{l}(w,h)}+t_{\hat{l}(w,h)+1} / 2.$
with probability that prediction is larger than ordinal value } k $= \mathcal{P}^{k}_{(w, h)} = P\left(\hat{\ell}(w, h) > k \,|\, \chi, \Theta \right) = (e^{\mathcal{Y}(w, h, 2k)}) / (e^{\mathcal{Y}(w, h, 2k)} + e^{\mathcal{Y}(w, h, 2k+1)})$


###### P3Dept
Exploit piecewise planarity. Predict plane coeffs $C(u,v)$ instead of depth $D(u,v)$, grouping pixels on the same 3D plane.  
**Plane eq**: $n\cdot P + d=0,\;n=(a,b,c)$, point $P=(X,Y,Z)$ from pinhole model $Z=D(u,v),\;X=\frac{Z(u-u_0)}{f_x},\;Y=\frac{Z(v-v_0)}{f_y}$.  
**Coeffs** $C=(\alpha,\beta,\gamma,\rho)$ with $\alpha=\frac{\hat{a}}{\rho},\beta=\frac{\hat{b}}{\rho},\gamma=\frac{\hat{c}}{\rho}$. Then $Z=[(\alpha\,u+\beta\,v+\gamma)\,\rho(u,v)]^{-1}$.  
**Offset** $\mathbf{o}(p)=q-p$ seeds from neighbors. **Resample** $C_s(p)=C[p+\mathbf{o}(p)]$.  
**Depth from seed** $D_s(u,v)=h\bigl(C_s(u,v),u,v\bigr)$.  
**Adaptive fusion** $D_f(u,v)=F(u,v)\,D_s(u,v)+(1-F(u,v))\,D_i(u,v)$.  
###### Monocular depth completion
First pass RGB and sparse depth (missings are filled with 0's) through a seperate layer, then concat the channels. Pass them through a deep regression network with skip connections. 
###### End-to-end learning of geometry and context for deep stereo regression
Built as Siamese network. Correlation layers concat. (along ch. dim.) left and right image feature tensor ($H \times W \times C$) with increasing horiz. shifts. The result is a 4D tensor with a new disparity dim. Then 3D conv. reduce the ch. dim. to 1 resulting in a ($H \times W \times \text{max. disp.} + 1$) tensor. Inerpreted as cost logit vectors, use **soft-argmin** to predict disparity per pixel. Train with $L_1$ loss on disparity prediction. 
**Siamese network loss:** $\min \| f(x^{(i)}) - f(x^{(j)}) \|, \text{ if similar; } \max \| f(x^{(i)}) - f(x^{(j)}) \|, \text{ otherwise.}$
**Soft-argmin:**  $\hat{d}[i, j] = \sum^{D_{max}}_{D=0}D*softmax(-C[i, j, D])$
###### Unsupervised depth estimation – left-right consistency
Use a CNN to predict the disparity of the left image. Then warp the pixels by predicted disparity. Take the difference to the true as the loss.  $u = (kf x^w) / z^w, \, v = (l f y^w) / z^w, \, u' = (kf (x^w - b)) / z^w, \, v' = (l f y^w) / z^w$  
###### Unsupervised depth estimation – ego motion from video
No stereo data, just video. CNN-2 predicts the pose of the previous and next frames cam. Synthesize views with a warping and use a photoconsistency loss (L1). 
$p_s \sim K \hat{T}_{t \to s} \hat{D}_t(p_t) K^{-1} p_t$  where $\hat{D}_t(p_t)$  is the predicted depth from CNN-1. 
##### Object Detection
###### Average Precision AP
For a fixed IoU Thr. and class compute the area under the prec-recall curve. 
Precision: TP out of all positives; Recall: TP out of all true objects
Average this over different IoU Thr. and all classes for mAP. 
###### R-CNN Series
**Fast R-CNN** introduces RoI pooling. Uses Huber loss  for robustness in regression. 
**Faster R-CNN** introduces RPN, sliding window of anchors with objectness score. Anchors  with IoU > 0.7 are pos. and <0.3 neg. during training.  First train RPN, then 2nd stage, then both together. $\mathcal{L}(\{p_i\}, \{t_i\}) = \frac{1}{N_{\text{cls}}} \sum_i \mathcal{L}_{\text{cls}}(p_i, p_i^*) + \lambda \frac{1}{N_{\text{reg}}} \sum_i p_i^* \mathcal{L}_{\text{reg}}(t_i, t_i^*)$  $N_{reg} = \#\text{pos}$
Use a FPN for scale invariance. 
###### Direct set prediction for object detection - DETR
Add position encoding to CNN image features, pass it trough encoder. The output is used in the decoder, which enriches $N$ object queries, which are then passed to a final FFN to predict class and box. Use the hungarian algo. to compute matching ground truth. 
$\mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)}) = -[c_i \neq \emptyset] \hat{p}_{\sigma(i)}(c_i) + [c_i \neq \emptyset] \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)})$ then use the matching for: 
$\mathcal{L}_{\text{Hungarian}}(y, \hat{y}) = \sum_{i=1}^{N} \left[ -\log \hat{p}_{\sigma(i)}(c_i) + [c_i \neq \emptyset] \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)}) \right]$
##### Instance Segmentation 
###### Mask R-CNN
Add decoupled FCN to predict masks. Uses RoI Align not RoI Pool. Adds a binary-ce loss.  $\mathcal{L}_\text{mask}(s^u, w) = -\frac{1}{m^2} \sum_{i=1}^{m^2} \Big( (1 - w[i]) \log(1 - s^u[i]) + w[i] \log(s^u[i]) \Big)$ 
Low resolution, slow, and needs non-max-suppression. 


# TODO spatial emebeddings
###### Joint optimization of spatial embeddings and clustering bandwidth
Assign each pixel $x_i$ to an instance $S_k$ by learning offsets to the centroid $C_k=\frac{1}{|S_k|}\sum_{x\in S_k}x$.  
**Naive**: direct centroid regression $\mathcal{L}_\text{regr}=\sum_i\lVert o_i-(C_k-x_i)\rVert^2$.  
**Joint opt**: spatial embedding $e_i=x_i+o_i$, bandwidth $\sigma_i$. Compute $\phi_k(e_i)=\exp\!\bigl(-(\lVert e_i-C_k\rVert^2) / (2\,\sigma_k^2)\bigr)$ with $\sigma_k=\frac{1}{|S_k|}\sum_{x_i\in S_k}\sigma_i$.  
**Assignment**: $x_i$ to $S_k$ if $\phi_k(e_i)>0.5$.   **Sequential clustering**: (i) pick highest seed as new centroid $(\hat{C}_k,\hat{\sigma}_k)$ (ii) assign all $x_i$ satisfying condition to instance (iii) repeat.   **Supervision**: $\phi_k$ supervised via Lovasz-hinge. Offsets $o_i$ and $\sigma_i$ supervised implicitly by backprop.




##### Panoptic Segmentation - class and instance labels
**Panoptic Quality** First match segments (IoU greater 0.5)  for a single class: $PQ=\frac{\sum_{(p,g)\in\text{TP}}\mathrm{IoU}(p,g)}{|TP|+\tfrac12|FP|+\tfrac12|FN|} = \underbrace{\tfrac{\sum_{\text{TP}}\mathrm{IoU}(p,g)}{|TP|}}_{\text{SQ}}\times\underbrace{\tfrac{|TP|}{|TP|+\tfrac12|FP|+\tfrac12|FN|}}_{\text{RQ (F1)}}.$  
For uncountable classes, one GT segment per image, IoU computed image-wise then averaged. RQ = F1 score. SQ is the average IoU over matched segments.
###### Baseline: merge semantic and instance segmentation predictions
Merge sem. and instance seg. Resolve overlaps in favor of instances. 
###### Panoptic FPN
Add semseg branch to Mask R-CNN. Higher confidence wins overlap. Instance wins over stuff. Stuff smaller than threshold is removed. Segregations after post-processing!
###### MaskFormer
Use hungarian to define ground truth using binary mask overlap loss $\mathcal{L}_{\text{match}}(z_{\sigma(i)}, z_i^{gt}) = -[c_i^{gt} \neq \emptyset] p_{\sigma(i)}(c_i^{gt}) + [c_i^{gt} \neq \emptyset] \mathcal{L}_{\text{mask}}(m_{\sigma(i)}, m_i^{gt})$
$\mathcal{L}_{\text{mask-cls}}(z, z^{gt}) = \sum_{i=1}^N -\log(p_{\hat{\sigma}(i)}(c_i^{gt})) + [c_i^{gt} \neq \emptyset] \mathcal{L}_{\text{mask}}(m_{\hat{\sigma}(i)}, m_i^{gt})$ with this as the final loss. 
CNN encoder-decoder, the mid features are passed to the $N$ mask query encoder. Decoder output is fed to MLP for class prediction, and for mask embeddings. The dot product of mask and per-pixel embed. predicts the mask. For panop. and instance **inference** $\arg \max_{i: c_i \neq \emptyset} p_i(c_i) \cdot m_i[h, w]$ For semseg predict $\arg \max_{c \in \{1, \dots, K\}} \sum_{i=1}^N p_i(c) \cdot m_i[h, w]$ 
**Mask2Former**: 1. mask the decoder cross-attention 2. do cross-atte. before self-atte. 3. pass higher res features to decoder 4. Approximate $\mathcal{L}_{mask}$ with uniform and importance sampling. **OneFormer**: one model trained for all 3 tasks. Adds a task token and a task text. 
# TODO OneFormer
 
 ##### Unimodal 3D Object Detection
**Multi-View CNN**: do element wise max pooling of feature maps extracted from renderings of different perspectives and feed to CNN-2.  **3D CNN**: use oct-tree
###### Point Net
**1.** predicted rigid body transform (RBT) points **2.** individual MLP (to higher dim. channels) **3.** another predicted RBT **4.** another individual MLP **5.** max-pool over channel dimension, produces one global feature vector (symmetric fn.) **6.** final MLP predicts logits
**SemSeg**: concat local per point features, before pooling with global vector and predict logits for every point. 
###### PointNet++
**SetAbstraction**: **1.** iterative farthest point sampling **2.** Group points within a ball around the sampled anchors **3.** PointNet layer (individual MLP, max pool across channels, MLP on global output vector) **SemSeg**: alternate interpolation plus skip link concatenation, and individual per point MLP. 
**Interpolation:** $w_{ij} = 1 / d(\mathbf{x}_j, \mathbf{x}_i)^p, \; i = 1, \dots, k$ (KNN)  $\mathbf{f}(\mathbf{x}_j) = \sum_{i=1}^k w_{ij} \mathbf{f}(\mathbf{x}_i))/(\sum_{i=1}^k w_{ij})$   
###### VoxelNet - LiDAR
**Voxelization**: **1.** divide pcd **2.** sample t points per voxel **3.** sample's 3D position and centroid offset  and RRP makeup feature vector **4.** pass through individual MLP **5.** max pool over channel produces, a single voxel feature vector.  The sparse 4D tensor is passed to the **Conv. middle layer**: 3D conv. BN, ReLU. Until only BEV, which is is passed to the **RPN**: Faster-RCNN style U-net, predicts anchor probability and
**BBox regression**: Residual $u^*=(\Delta x,\Delta y,\Delta z,\Delta l,\Delta w,\Delta h,\Delta\theta)$ with $d_a=\sqrt{(l^a)^2+(w^a)^2}.$  
$\Delta x=\frac{x_c^g - x_c^a}{d_a},\;\Delta y=\frac{y_c^g - y_c^a}{d_a},\;\Delta z=\frac{z_c^g - z_c^a}{h_a},\;\Delta l=\log\!\bigl(\frac{l^g}{l^a}\bigr),\;\Delta w=\log\!\bigl(\frac{w^g}{w^a}\bigr),\;\Delta h=\log\!\bigl(\frac{h^g}{h^a}\bigr),\;\Delta\theta=\theta^g-\theta^a,$  
**VoxelNet loss** $L=\alpha\,\frac{1}{N_\text{pos}}\sum_i L_\text{cls}(p_i^\text{pos},1)+\beta\,\frac{1}{N_\text{neg}}\sum_j L_\text{cls}(p_j^\text{neg},0)+\frac{1}{N_\text{pos}}\sum_i L_\text{reg}(u_i,u_i^*)$.  
Smooth L1 (Huber) for reg; cross-entropy for cls. (Augment with synthetic 3D obj.)
###### PointRCNN 
Use PointNet++ for per point features, classify into background or foreground. For foreground do **bin based 3D box regres.** discrete x-y gird with origin at point, classify the bin containing the box center and yaw angle. z and box dim. use normal regression. 
Bin length: $\delta = (2S) / C$     Half-side of search space: $\mathcal{S}$    Number of bins $\mathcal{C}$
$\text{bin}_x^{(p)} = \Bigl\lfloor (x^p - x^{(p)} + S) / \delta \Bigr\rfloor, \quad \text{bin}_y^{(p)} = \Bigl\lfloor (y^p - y^{(p)} + S) / \delta \Bigr\rfloor$  
$\text{res}_u^{(p)} = (1 / C) \Bigl(u^p - u^{(p)} + S - \bigl(\text{bin}_u^{(p)} \cdot \delta + (\delta / 2)\bigr)\Bigr), \quad \text{res}_z^{(p)} = z^p - z^{(p)}$
**Loss** for each positive anchor $p$:   $L_\text{bin}^{(p)}=\sum_{u\in\{x,y,\theta\}}\Bigl[F_\text{cls}\bigl(\text{bin}_u^{(p)},\,\widehat{\text{bin}_u^{(p)}}\bigr)+F_\text{reg}\bigl(\text{res}_u^{(p)},\,\widehat{\text{res}_u^{(p)}}\bigr)\Bigr],\quad L_\text{res}^{(p)}=\sum_{v\in\{z,h,w,l\}}F_\text{reg}\bigl(\text{res}_v^{(p)},\,\widehat{\text{res}_v^{(p)}}\bigr).$  
Overall: $L_\text{reg}=\tfrac{1}{N_\text{pos}}\sum_{p\in\text{pos}}\bigl[L_\text{bin}^{(p)}+L_\text{res}^{(p)}\bigr].$  
**Final prediction** augment proposed BBox points with fore-backgroud prediction and per point feature vectors. Transform to canonical frame and feed to PointNet++ for bin-based refinement and confidence prediction. 
##### 3D Reconstruction and Localization

# TODO ICP

**Epipolar constraint** $p_2^T E p_1 = 0$  with **essential matrix** $E = [t]_{\times}R$ 
###### SuperGlue
**Attentional aggregation**: start with descriptors (e.g. SIFT) from both images, add positional/confidence embeddings via a small MLP. Pass these per-image features through $L$ blocks, each with: **1. Self-attention** (features attend to others in the same image):  
   $^{(l)}\mathbf{Q}^{(1)} = ^{(l)}\mathbf{W}_Q \mathbf{X}^{(l-1)}(1)$, etc. with $\text{softmax}\bigl(^{(l)}\mathbf{Q}^{(1)\top} \, ^{(l)}\mathbf{K}^{(1)}\bigr) \, ^{(l)}\mathbf{V}^{(1)}.$  
**2. Cross-attention** (features attend to the other image):  
   $^{(l)}\mathbf{Q}^{(1)} = ^{(l)}\mathbf{W}_Q \mathbf{X}^{(l-1)}(1)$, etc. with $\text{softmax}\bigl(^{(l)}\mathbf{Q}^{(1)\top} \, ^{(l)}\mathbf{K}^{(2)}\bigr) \, ^{(l)}\mathbf{V}^{(2)}.$  
After $L$ blocks, a final linear projection yields descriptors $f_i^{(1)}, f_j^{(2)}$ for **Differentiable optimal matching**: start with dot product between features as score, add dustbin to build a score matrix and normalize using the Sinkhorn algorithm. Train with $L_{matching} = -\sum_{(i, j) \in M}log(P_{ij}$)
###### NeRF
MLP mapping (3D pos. $\mathbf{x}$, viewing direction $\theta, \phi$) $\rightarrow$ (density $\sigma$, RGB color $\mathbf{c}$).
MLP-1 maps $\mathbf{x} \rightarrow (\mathbf{f}, \sigma)$  and MLP-2 maps $(\mathbf{f}, (\theta, \psi)) \rightarrow \mathbf{c}$    
**Volume rendering** approximate color along ray $r=o+t\,d$:  
$\hat{C}(r)=\sum_{i=1}^N T_i\bigl(1-\exp(-\sigma_i\,\delta_i)\bigr)\,c_i,\quad T_i=\exp\Bigl(-\sum_{j=1}^{i-1}\sigma_j\,\delta_j\Bigr).$  
Render loss is $\|\hat{C}(r)-C_{\text{gt}}(r)\|^2$.  
**Positional encoding** map coordinates to higher frequency:  
$\gamma(p)=\bigl(\sin(2^0\pi p),\,\cos(2^0\pi p),\,\dots,\sin(2^{L-1}\pi p),\,\cos(2^{L-1}\pi p)\bigr).$  
**NeRF in the wild** splits scene into: **Static** density $\sigma$, color that can depend on an input appearance embedding. **Transient** density/color for dynamic objects, controlled by a transient embedding.  
**Block-NeRF**: Splits large driving scenes into sub-NeRFs (e.g., per intersection). **Visibility Predictor:** Each sub-NeRF uses fv(x,d)f_v(x, d) to model transmittance. **Training:** Masks dynamic objects via semantic segmentation. **Inference:** Combines sub-NeRF outputs with distance-based weights wi∼d(c,xi)−pw_i \sim d(c, x_i)^{-p} and visibility. **Transition Handling:** Discards sub-NeRFs with low mean visibility for smooth transitions.
##### Domain adaptation
**Unsupervised DA** Source: labeled data  Target: unlabeled data
**Weakly supervised DA** Source: labeled data Target: unlabeled data with correspondences
**Source-free DA (model adaption)** Source: pretrained model  Target: unlabeled data
**Test-time or online DA** Source: pretrained model Target: unlabeled sequential test data
**Domain generalization** Source: labeled data  Target: none, no adaptation at test time
###### Input level DA
**Fog simulation** **1.** compute depth estimate (via plane fitting, outlier filtering) **2.** transmittance $\tilde{t}(x)=exp(-\alpha\,\ell(x))$ **3.** apply fog model $I(x)=R(x)\,\tilde{t}(x)+L\bigl(1-\tilde{t}(x)\bigr)$.
Atmospheric light $L$, attenuation coeff. $\alpha$ Can be enhanced with bilateral filtering and color and semantics from labels.  $\mathbf{t}(\mathbf{p}) = \frac{\sum_{\mathbf{q} \in \mathcal{N}(\mathbf{p})} G_{\sigma_s}(\|\mathbf{q} - \mathbf{p}\|) \bigl[\delta(h(\mathbf{q}) - h(\mathbf{p})) + \mu G_{\sigma_c}(\|\mathbf{R}(\mathbf{q}) - \mathbf{R}(\mathbf{p})\|)\mathbf{t}(\mathbf{q})\bigr]}{\sum_{\mathbf{q} \in \mathcal{N}(\mathbf{p})} G_{\sigma_s}(\|\mathbf{q} - \mathbf{p}\|) \bigl[\delta(h(\mathbf{q}) - h(\mathbf{p})) + \mu G_{\sigma_c}(\|\mathbf{R}(\mathbf{q}) - \mathbf{R}(\mathbf{p})\|)\bigr]}$
**Rain simulation** Uses fog-like attenuation:  $t(x)=e^{-0.312\,R^{0.67}\,\ell(x)}$ (rainfall rate $R$ in mm/h, distance $\ell(x)$). Then $I_{att}(x)=R(x)\,t(x)+L\bigl(1-t(x)\bigr)$.   **Rain streak photometry**: $S'=S\bigl(0.94\,\mathbf{F}+0.06\,\mathbf{E}\bigr)$, combining refraction/reflection from environment map.  
Alpha compositing: $\mathbf{I}_{\text{rain}}(\mathbf{x}) = \mathbf{I}_{\text{att}}(\mathbf{x}) \frac{T - S'_\alpha(\mathbf{x})T}{T} + S'(\mathbf{x})$ (exposure $T$, streak duration $\tau$).
**LiDAR Snowfall** $P_{R,\text{snow}}(R) = P_{R,\text{snow}}^0(R) + \sum_{j=1}^n P_{R,\text{snow}}^j(R)$
**CycleGAN:** Train two conditional GANs $F: S \rightarrow T$ and $G: T \rightarrow S$ add  $|| G(F(x)) -  x ||$ to  optimization **FDA** replaces low-freq amplitude of the source with that of the target. Define a low-pass mask $M_\beta(h,w)=1_{(h,w)\in[-\beta H:\beta H]\times[-\beta W:\beta W]}$. Then  
$I_{s\to t}=\mathcal{F}^{-1}\bigl(\bigl(M_\beta\,\mathcal{F}^A(I_t)+(1-M_\beta)\,\mathcal{F}^A(I_s)\bigr),\,\mathcal{F}^P(I_s)\bigr).$
###### Feature level DA 
**CyCADA**: models $f_S, f_T$   generators $G_{S \rightarrow T}, G_{T \rightarrow S}$   discriminators $D_T, D_{feat}$ 
$\mathcal{L}_{\text{GAN}} = \mathbb{E}_{I_t \sim X_T}[\log D_T(I_t)] + \mathbb{E}_{I_s \sim X_S}[\log(1 - D_T(G_{S \to T}(I_s)))]$
$\mathcal{L}_{\text{cyc}} = \mathbb{E}_{I_s \sim X_S}[\| G_{T \to S}(G_{S \to T}(I_s)) - I_s \|_1] + \mathbb{E}_{I_t \sim X_T}[\| G_{S \to T}(G_{T \to S}(I_t)) - I_t \|_1]$
$\mathcal{L}_{\text{GAN,feat}} = \mathbb{E}_{I_t \sim X_T}[\log D_{\text{feat}}(f_T(I_t))] + \mathbb{E}_{I_s \sim X_S}[\log(1 - D_{\text{feat}}(f_T(G_{S \to T}(I_s))))]$
**CISS**: train images $(I_S, I_T)$ and stylized versions $(I_{S \rightarrow T}, I_{S \rightarrow T})$ are passed through siamese network.  $\mathcal{L}_{\text{inv}}(F, I, I') = \frac{1}{DMN} \|\phi(I) - \phi(I')\|_F^2$  
$\mathcal{L}_{\text{CISS}} = \mathcal{L}_{\text{CE}}(F, I_s, Y_s) + \mathcal{L}_{\text{CE}}(F, I_t, \hat{Y}_t) + \lambda_s \mathcal{L}_{\text{inv}}(F, I_s, I_{s \to t}) + \lambda_t \mathcal{L}_{\text{inv}}(F, I_t, I_{t \to s})$
**CMA (source free, unlabeled S-T pairs)**: extract features of S-T pair with ENC. Align patches of S-T pair, average over features for patch features. An S patch is the anchor, the matching T patch the positive.  $\mathcal{L}_{\text{cdc}, i} = -\log \frac{\exp(\mathbf{a}_i^\top \mathbf{p}_i / \tau)}{\exp(\mathbf{a}_i^\top \mathbf{p}_i / \tau) + \sum_{j=1}^M \exp(\mathbf{a}_i^\top \mathbf{n}_j / \tau)}$
$\mathcal{L}_{\text{cdc}} = \frac{\sum_i \mathcal{L}_{\text{cdc}, i} [\bar{c}_i \geq 0.2]}{\sum_i [\bar{c}_i \geq 0.2]}$ use only patches with avg. confidence > 0.2 
###### Output level adaptation
**AdaptSegNet**: CyCADA but with per pixel output discriminator.  $\mathcal{L}_{\text{adv}, D} = \mathbb{E}_{I_s \sim X_S} \left[ \sum_{h, w} \log D(P_s)(h, w) \right] + \mathbb{E}_{I_t \sim X_T} \left[ \sum_{h, w} \log \left( 1 - D(P_t)(h, w) \right) \right]$
$\mathcal{L}_{\text{adv}, f} = \mathbb{E}_{I_t \sim X_T} \left[ \sum_{h, w} \log \left( 1 - D(P_t)(h, w) \right) \right]$

# TODO self training

**DACS**: create artificial mixed (S-T) images for training. 
**Probabilistic learned warping for output alignment**: from images $I, J$ generate $I'$ with $W^{-1}$. Predict $\mathbf{F_{I'\rightarrow I}}, \mathbf{F_{J\rightarrow I}}, \mathbf{F_{I\rightarrow J}}$  Train: $\mathcal{L}_{I \to I} = \| \hat{\mathbf{F}}_{I \to I} - \mathbf{W} \|^2$
$\mathcal{L}_{I \to J \to I} = \| \mathbf{V} \cdot (\hat{\mathbf{F}}_{I \to J} + \Phi \hat{\mathbf{F}}_{J \to I} (\hat{\mathbf{F}}_{J \to I}) - \mathbf{W}) \|^2 = \| \mathbf{V} \cdot (\hat{\mathbf{F}}_{I \to J \to I} - \mathbf{W}) \|^2$
**UAWarpC** models per-pixel 2D warps with heteroscedastic Gaussian uncertainty:  
$p(\mathbf{W}\mid I,I')=\mathcal{N}\bigl(\mathbf{W};\hat{\mathbf{F}}_{I'\to I},\hat{\Sigma}_{I'\to I}\bigr),\quad p(\mathbf{W}\mid I,J,I')=\mathcal{N}\bigl(\mathbf{W};\hat{\mathbf{F}}_{I'\to J\to I},\hat{\Sigma}_{I'\to J\to I}\bigr).$  
**Warp net** is trained by maximizing warp log-likelihood. A pixel-level **confidence** $C(I_1,I_2)=1-\exp\!\bigl(-r^2/2\,\sigma_{I_2\to I_1}^2\bigr)$ helps downweight non-corresponding regions and trust high-uncertainty predictions less.  
###### Test-time DA - TENT Adaptively compute the mean and variance for BA over the test batch. 
##### Multi-modal Object Detection
**Frustum PointNet**: reduce search space to 2D CNN box prediction
**MV3D**: project pcd to BEV and fuses with camera view
**PointPainting**: use semseg model and project semantic features onto points in the pcd
**MVX-Net**: use VGG OD features not semseg **Mid-level MVX-Net**: fusion at voxel stage, project voxels onto image, pool image features and concat with voxel feature.
**PointAugmenting**: uses more local feature and does input and mid voxel fusion
**TransFusion:** mid-level lidar-camera fusion with soft association via cross attention. Predict BEV heatmap, select top candidates as Quries. Max pool image features over column (collapse height), exract keys and values from it.   **SMCA**: project query position from BEV space to the 2D camera view: $(c_x, c_y)$. Multiply cross-attention map element-wise with a soft Gaussian mask $M_{ij} = \exp(\left(-(i - c_x)^2 + (j - c_y)^2) / \sigma r^2\right)$
**AVOD**: project 3D anchor box grid into input image and BEV input and RoI pool. (mid fustion). Then again combine proposals with BEV and image features and refine the top k candidates (late fusion) **CLOC (pure late fusion)**: $k$ 2D predictions and $n$ 3D predictions. When there is high IoU fill the entry of a $k \times n \times 4$ tensor with (IoU, the 2 confidences, distance to the box) the 4 dim features are processed by a MLP to 1 dim. Max pool over $k$ for the final $n$ confidences.
**CenterFusion**: take closest **radar** pillar which intersects with the frustum and concatenate image features with radial velocity (in x, y) and the depth from radar. 
$F_{x,y,i}^j = 1/ M_i \begin{cases} f_i, & |x - c_x^j| \leq \alpha w^j \text{ and } |y - c_y^j| \leq \alpha h^j \\ 0, & \text{otherwise} \end{cases}$
**CRF-Net: camera radar 2D detection**:  Project radar detection as 3 meter pillar into image, add channel with distance and cross section. The radar channels are added at different depths of the VGG encoder. Use **BlackIn** (deactivate image input with prob 0.2) to increase dependence on radar features. 
###### Robust fusion 
**Local patch-level measurement entropy**  
$p_i^{mn}=\frac{1}{MN}\sum_{j=1}^M\sum_{k=1}^N \delta\bigl(I(m+j,n+k)-i\bigr),\quad i=0\ldots255.$   $\rho^{mn}=\sum_{i=0}^{255}p_i^{mn}\log\bigl(p_i^{mn}\bigr).$  
Higher patch entropy $\rightarrow$ higher SNR.
**Deep entropy-based adaptive fusion**  
**1.** Compute entropy map for each sensor stream (LiDAR, camera, etc.).  
**2.** Convolve + sigmoid $\rightarrow$ weights in $[0,1]$.   **3.** Pointwise multiply concatenated features by entropy-derived weights (amplify/attenuate).  **4.** Concatenate initial entropy with scaled features for subsequent detection layers.
**HRFuser:** extends HRNet for multi-modality (camera primary, others additional).
Parallel branches for camera modality. Multi-Window Cross-Attention (**MWCA**) fuses features across modalities.