
# Multiple object tracking (MOT) 
In 2D track instances over frames, giving a unique Id to each instance which in consistent across frames. 

## Metrics 
1. **Metrics which highlight the different types of errors of an algorithm**
   - **Mostly Tracked (MT)** trajectories: number of ground-truth trajectories that are correctly tracked in at least 80% of the frames.
   - **Fragments**: trajectory hypotheses that cover at most 80% of a ground-truth trajectory.
   - **Mostly Lost (ML)** trajectories: number of ground-truth trajectories that are correctly tracked in less than 20% of the frames.
   - **False trajectories**: trajectory hypotheses which do not correspond to a real object.
   - **ID switches**: number of times when an object is correctly tracked, but the associated ID for the object is erroneously changed.

2. **Summarizing metric: MOTA**
   - Decide whether an object and a prediction match based on the Intersection over Union (IoU) of their bounding boxes.
   - If the object $o_i$ and the hypothesis $h_j$ are matched in frame $t-1$, and in frame $t$ we have $IoU(o_i, h_j) \geq 0.5$, then $o_i$ and $h_j$ are matched in frame $t$.
   - Remaining objects are matched with remaining hypotheses, based on the 0.5 IoU threshold.
   - **False negatives (FN)**: ground-truth boxes that are not associated with a hypothesis.
   - **False positives (FP)**: hypotheses that are not associated with a ground-truth box.
   - **ID switches (IDSW)**: number of times the ID of a tracked ground-truth object is incorrectly changed.

   **Multiple Object Tracking Accuracy (MOTA):**
   $$
   MOTA = 1 - \frac{FN + FP + IDSW}{GT}, \quad MOTA \in (-\infty, 1]
   $$

## Classical approach: tracking by detection
1. Detect objects (off the shelf detection)
2. Extract per object features
3. Affinity computation between detections
4. Associate detections
![[Pasted image 20250205111019.png]]


## Baseline 
1. Off the shelf object detector
2. Kalman filter for motion prediction
3. Hunagarian algorithm for assiciation


## Siamese CNNs for MOT
Use a siamese CNN to predict a matching score between two boxes. 
For training, use positive samples from ground truth, and negatives from false positives or different ground truth trajectories. 

It solves the association as a max flow problem. 

$\mathcal{D} = \{d_i^t : d_i^t = (x, y, t)\}$ : set of detections across the entire input video

$T_k = \{d_{k_1}^{t_1}, d_{k_2}^{t_2}, \ldots, d_{k_N}^{t_N}\}$ : a trajectory as an ordered list of detections

**Optimization**: identify the set of trajectories $\mathcal{T}^* = \{T_k\}$ which best explains the given detections.

**Linear programming – max flow**

$$
\mathcal{T}^* = \arg \min_{\mathcal{T}} \sum_i C_{\text{in}}(i) f_{\text{in}}(i) 
+ \sum_i C_{\text{out}}(i) f_{\text{out}}(i) 
+ \sum_i C_{\text{det}}(i) f(i) 
+ \sum_{i, j} C_{\text{t}}(i, j) f(i, j)
$$

**Where:**

$$
C_{\text{det}}(i) =
\begin{cases} 
-\frac{s_i}{V_{\text{det}}} + 1 & \text{if } s_i < V_{\text{det}} \\ 
-\frac{s_i + 1}{1 - V_{\text{det}}} - 1 & \text{if } s_i \geq V_{\text{det}}
\end{cases}
$$

$$
C_{\text{t}}(i, j) =
\begin{cases} 
-\frac{s_{i, j}^{\text{RF}}}{V_{\text{link}}} + 1 & \text{if } s_{i, j}^{\text{RF}} < V_{\text{link}} \\ 
-\frac{s_{i, j}^{\text{RF}} + 1}{1 - V_{\text{link}}} - 1 & \text{if } s_{i, j}^{\text{RF}} \geq V_{\text{link}}
\end{cases}
$$

- Constraints: flow conservation at the nodes, exclusion constraints.
- The **in/out costs** are positive and are used so that the tracker does not indiscriminately create too many trajectories.
- **Matching scores** from Siamese affinity computation.
- Negative costs ensure that if all were positive, zero flow = no trajectory would be the trivial solution.

### Limitations
* detection is based on a single frame
* association is based on heuristics in post processing
* minimal temporal modeling


## End-to-end framework for MOT with transformers (MOTR)
Inspired by [[6 - Object Detection#Detection Transformer (DETR)]] but in DETR we had object queries, now we query for trajectories/tracks. 

![[Pasted image 20250205143129.png]]

Now the transformer decoder has detect queries and track queries. 
When a detect query confidence is high enough a fresh detection is born and then added as a track query in the next frame. 
During training fresh detections are added to the tracks based on ground truth. 


# Motion prediction
Birds eye view for cars. 

## CNN based prediction
Pass BEV rendered image to CNN with car of interest in the center
To handle temporal data: 

* use a RNN decoder. 
![[Pasted image 20250205145149.png]]

* render past data 
![[Pasted image 20250205145249.png]]

* late fusion with past state
![[Pasted image 20250205145347.png]]

* input multiple frames
![[Pasted image 20250205145432.png]]


## RNN based
![[Pasted image 20250205150906.png]]


