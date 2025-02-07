

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