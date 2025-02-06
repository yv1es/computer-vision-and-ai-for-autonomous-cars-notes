# Taxonomy

**Automated driving system (ADS):** The hardware and software that are collectively capable of performing the **entire dynamic driving task** on a sustained basis, regardless of whether it is limited to a specific operational design domain.

**Driving automation system:** The hardware and software that are collectively capable of performing **part or all of the dynamic driving task** on a sustained basis.

$$
\text{Automated driving systems} \subseteq \text{Driving automation systems}
$$


# SAE's levels of driving automation
Level 0 means no automation. 

**Driving automation systems are categorized into levels** based on:
1. Whether the driving automation system (DAS) performs either the longitudinal or the lateral vehicle motion control subtask of the dynamic driving task (DDT). ("Feet off")
2. Whether the DAS performs both the longitudinal and the lateral vehicle motion control subtasks of the DDT simultaneously. ("Hands off")
3. Whether the DAS also performs the OEDR subtask of the DDT. ("Eyes off")
4. Whether the DAS also performs DDT fallback. ("Mind off")
5. Whether the DAS is limited by an operational design domain.



# Etc
**Sigmoid** $\sigma(x) = \frac{1}{1 + exp(-x)}$  
**Softmax** $softmax(z)_i = \frac{exp(z_i)}{\Sigma_jexp(z_j)}$ 
**SGD with momentum** ($\alpha$  slightly below 1): $\theta \leftarrow \theta + \mathbf{v}$ with
$$\mathbf{v} \leftarrow \alpha \mathbf{v} - (1 - \alpha) \eta \nabla_\theta \left( \frac{1}{B} \sum_{i=1}^B \mathcal{L}(f(x_i; \theta), y_i) \right) $$

**Kaiming initialzation:** 0-mean Gaussian with stddev = $\sqrt{2/d_{l-1}}$ 
**Batch Norm. (BN):** First each unit is normalized (subtract mean and divide by variance of the mini-batch). Then secondly scale (by $\gamma$) and shift (by $\beta$) (learnable $\gamma, \beta$).  
**Conv. output dimension:** $\lfloor \frac{H + 2P - m}{s} \rfloor + 1$   
**Attention:** $X_l = \text{softmax}(Q_l K_l^\top) V_l + X_{l-1}$ 
**Masked:** $X_l = \text{softmax}(\mathcal{M}_{l-1} + Q_l K_l^\top) V_l + X_{l-1}$  w. $\mathcal{M}_{l-1}(x, y) = \begin{cases} 0 & \text{if } \mathbf{M}_{l-1}(x, y) = 1 \\ -\infty & \text{otherwise} \end{cases}$

