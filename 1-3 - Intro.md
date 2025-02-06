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
Sigmoid $\sigma(x) = \frac{1}{1 + exp(-x)}$  
Softmax $softmax(z)_i = \frac{exp(z_i)}{\Sigma_jexp(z_j)}$ 

SGD with momentum ($\alpha$  slightly below 1) 
![[Pasted image 20250206084412.png]]

Kaiming initialzation: 0-mean Gaussian with stddev = $\sqrt{2/d_{l-1}}$ 

Batch Norm. (BN): First each unit is normalized (subtract mean and divide by variance of the mini-batch). Then secondly scale (by $\gamma$) and shift (by $\beta$) (learnable $\gamma, \beta$).  

Conv. output. $\lfloor \frac{H + 2P - m}{s} \rfloor + 1$   



