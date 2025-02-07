

##### Etc
**Compute mean** $\mu_j=\frac{1}{N}\sum_{i=1}^N x_{ij}$  
**Compute stdev** $\sigma_j=\sqrt{\frac{1}{N-1}\sum_{i=1}^N\bigl(x_{ij}-\mu_j\bigr)^2}$  
**Sigmoid** $\sigma(x) = \frac{1}{1 + exp(-x)}$  
**Softmax** $softmax(z)_i = \frac{exp(z_i)}{\Sigma_jexp(z_j)}$ 
**Cross Entropy Loss:** $-\Sigma_{x \in classes} y(x)log(p(x))$ 
**SGD with momentum** ($\alpha$  slightly below 1): $\theta \leftarrow \theta + \mathbf{v}$ with
$$\mathbf{v} \leftarrow \alpha \mathbf{v} - (1 - \alpha) \eta \nabla_\theta \left( \frac{1}{B} \sum_{i=1}^B \mathcal{L}(f(x_i; \theta), y_i) \right) $$
**Kaiming initialzation:** 0-mean Gaussian with stddev = $\sqrt{2/d_{l-1}}$ 
**Batch Norm. (BN):** First each unit is normalized (subtract mean and divide by variance of the mini-batch). Then secondly scale (by $\gamma$) and shift (by $\beta$) (learnable $\gamma, \beta$).  
**Conv** $y[p,q]=\sum_{i=1}^m\sum_{j=1}^n w[i,j]\,x[p+i-1,q+j-1],\;p=1\ldots H-m+1,\;q=1\ldots W-n+1$  
**T. Conv** $y[p,q]=\sum_{i\in I}\sum_{j\in J}x[i,j]\,w[p-i+1,q-j+1],\;p=1\ldots H+m-1,\;q=1\ldots W+n-1$  
**Conv. output dimension:** $\lfloor \frac{H + 2P - k}{s} \rfloor + 1$   
**Attention:** $X_l = \text{softmax}(Q_l K_l^\top) V_l + X_{l-1}$   
**Masked:** $X_l = \text{softmax}(\mathcal{M}_{l-1} + Q_l K_l^\top) V_l + X_{l-1}$  w. $\mathcal{M}_{l-1}(x, y) = \begin{cases} 0 & \text{if } \mathbf{M}_{l-1}(x, y) = 1 \\ -\infty & \text{otherwise} \end{cases}$
**berHu Loss** $B(x)=\begin{cases}|x|,&|x|\le c\\ \tfrac{x^2+c^2}{2c},&|x|>c\end{cases}$ with $c=\tfrac{1}{5}\max_i|\tilde{y}_i - y_i|$.  
**Polar**: $\begin{bmatrix} x \\ y \\ z \end{bmatrix} = \begin{bmatrix} r \cos \alpha \cos \epsilon \\ r \sin \alpha \cos \epsilon \\ r \sin \epsilon \end{bmatrix}$ elevation $\epsilon$  azimuth $\alpha$ 
##### Self Driving Levels
**1** Feet off   **2** Hands off  **3** Eyes off  **4** Mind off  **5** Mind off, No ODD restriction 
##### LiDAR
**Sinusoidal Pulse** $P_T(t)=\begin{cases}P_0\,\sin^2\!\bigl(\frac{\pi t}{2\,\tau_H}\bigr),&0\le t\le2\,\tau_H\\0,&\text{otherwise}\end{cases}$  
**Square-Exponential Pulse** $P_T(t)=\begin{cases}C\,P_0\,\bigl(\frac{t}{\tfrac{\tau_H}{1.75}}\bigr)^2\,\exp\!\bigl(-\tfrac{t}{\tfrac{\tau_H}{1.75}}\bigr),&t\ge0\\0,&t<0\end{cases}$  
**Impulse Response** $H_T(R)=\rho_0\,\delta\bigl(R-R_0\bigr)$ clear weather 
**Received Power** $P_R(R)=C_A\,\frac{P_0\,\rho_0}{R_0^2}\,\sin^2\!\bigl(\frac{\pi\,(R-R_0)}{c\,\tau_H}\bigr),\,R_0\le R\le R_0+c\,\tau_H;\,0,\text{otherwise.}$  
**Symbols** $\tau_H:$ half-power width, $P_0:$ peak power, $c:$ speed of light, $\rho_0:$ diff. refl., $R_0:$ range, $C_A:$ const, $C:$ scale.
##### Radar
**Radar equation** $P_\text{received}=\frac{P_\text{transmitted}\,G\,\sigma\,A_e}{16\,\pi^2\,r^4\,L}$  
$G:$ gain, $\sigma:$ radar cross section, $A_e:$ aperture, $L:$ loss factor
#### Depth prediction
**Make3D** superpixels as planar via MRF plane coeffs.  
##### Metrics
**Threshold**: % of pixels s.t. $\max(\tfrac{y_i}{y_i^*},\tfrac{y_i^*}{y_i})=\sigma<\text{thr}.$  (common $thr.$ is $1.25^i$)
**Abs rel diff**: $\text{rel}=\tfrac{1}{T}\sum_i \tfrac{|\,y_i - y_i^*\,|}{y_i^*}.$   **Sq rel diff**: $\text{srel}=\tfrac{1}{T}\sum_i \tfrac{|\,y_i - y_i^*\,|^2}{y_i^*}.$  
**RMS**: $\sqrt{\tfrac{1}{T}\sum_i (\,y_i-y_i^*\,)^2}.$   **RMS(log)**: $\sqrt{\tfrac{1}{T}\sum_i|\log(y_i)-\log(y_i^*)|^2}.$  
##### Ordinal Regression
**Ordinal Loss**  $L(x,\Theta)=-\tfrac{1}{N}\sum_{w=0}^{W-1}\sum_{h=0}^{H-1}\Bigl[\sum_{k=0}^{l(w,h)-1}\log(\mathcal{P}^{k}_{(w, h)})+\sum_{k=l(w,h)}^{K-1}\log\bigl(1-\mathcal{P}^{k}_{(w, h)}\bigr)\Bigr].$
**Inference**  $\hat{l}(w,h)=\sum_{k=0}^{K-1}\eta\bigl(\mathcal{P}^{k}_{(w, h)}\ge0.5\bigr),\;\hat{d}(w,h)=t_{\hat{l}(w,h)}+t_{\hat{l}(w,h)+1} / 2.$
with probability that prediction is larger than ordinal value } k $= \mathcal{P}^{k}_{(w, h)} = P\left(\hat{\ell}(w, h) > k \,|\, \chi, \Theta \right) = (e^{\mathcal{Y}(w, h, 2k)}) / (e^{\mathcal{Y}(w, h, 2k)} + e^{\mathcal{Y}(w, h, 2k+1)})$

