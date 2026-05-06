# Fall Detection

This is my fall detection project, comparing performance between a temporal deep learning model (LSTM) and a rule-based model, both tuned and evaluated on the [FallVision](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/75QPKK) dataset (Rahman et al., 2025). 

## Overview

The project takes raw fall and no-fall videos from FallVision, comprising of bed, chair, and standing scenarios. Videos are resampled to 30fps, skeletal keypoints sequences are extracted from each frame, and these sequences are then segmented into 60-frame clips. The data is then used to tune both models: the LSTM is trained on the train set and then evaluated on the validation set to inform the next iterations and changes made, while the rule-based classifier is evaluated on the validation set to inform rule and threshold changes for next iterations. An alternative rule-based model was developed but did not outperform the original baseline, this model can be found [here](alt-rule-based/alt-rule-based.ipynb). The best performing LSTM version ([v5](models/lstm_v5)) was compared to the [rule-based](rule-based.ipynb) model by evaluating both on the test set and comparing performance metrics.

## File & Folder Overview

[alt-rule-based](alt-rule-based) - contains the alternative rule-based model created
[data](data) - contains all data states (excluding videos due to file sizes)
[example-vid](example-vid) - contains an example video from the dataset used for demo purposes
[final-model](final-model) - contains the final LSTM model
[models](models) - contains all LSTM model iterations
[pose-extraction-demo](pose-extraction-demo) - contains demo code to demonstrate how pose extraction works
[pre-processing](pre-processing) - contains all pre-processing code files
[dataset.py](dataset.py) - creates DataLoader elements to feed data into models
[LSTM-analyse](LSTM-analyse.ipynb) - LSTM analysis code to create graphs
[LSTM-train](LSTM-train.ipynb) - LSTM training code
[model-test](model-test.ipynb) - test set evaluation results and final model comparisons and graphs
[rule-based](rule-based.ipynb) - rule-based model code
[yolo11n-pose](yolo11n-pose.pt) - YOLO11-pose model for keypoint extraction


## Pose-Extraction

To convert the videos to numerical data that the models can process, [YOLO11-pose](https://docs.ultralytics.com/models/yolo11/) was used to extract skeletal keypoints from each frame in a video. An example of this is available [here](pose-extraction-demo/pose-extraction-demo.ipynb). YOLO11-pose detects 17 skeletal keypoints using the COCO keypoint format, extracting coordinates and a confidence score for each keypoint, per frame. An example of the output can be seen [here](pose-extraction-demo/pose_keypoints_frame1.csv). The example uses a video from the dataset saved in [example-vid](example-vid) for demo purposes.

## Pre-Processing

All pre-processing files can be found [here](pre-processing). 

The FallVision dataset consists of ~11,700 videos, however half of these are masked (have a YOLO pose estimation "overlay"), and the other half are raw. For this project, only the raw videos were used, as a more recent version of YOLO was available and therefore used to extract the most accurate keypoints possible. However, contrary to the claim in the paper, the videos are not a uniform FPS, but rather have varying frame rates (the code used to check fps distribution can be found [here](pre-processing/check_fps_distribution.ipynb)). In order to handle this, only videos at 24fps or higher were used and were resampled to 30fps, while the rest of the videos were discarded along with any corrupted ones (this process was done [here](pre-processing/resample_fps.ipynb)). 
Keypoint sequences were then extracted from each video using [keypoint-extraction.ipynb](pre-processing/keypoint-extraction.ipynb) ([output](data/extracted_keypoints)), and were then segmented into 60-frame (2 seconds) clips [here](pre-processing/clip-segmentation.ipynb) ([output](data/segmented_clips)). 
The dataset split was then defined by creating a JSON file assigning clips to each split, ensuring clips from the same video were kept together (the code for this process is available [here](pre-processing/set-split.ipynb), and an output is available [here](data/dataset_split.json)).

All data stages (excluding videos) are available [here](data), including v2 data used for an ablation study (explained later).

## Data-Loading

In order to load the data into the models, PyTorch DataLoaders were used to load each dataset split into a separate element by creating Dataset elements for each of the splits, and putting those into respective DataLoader elements. This also allowed for simple management of dataset split usage, and for simple-to-use functions for each of the splits. Throughout the project, the data was always normalised before usage to eliminate resolution dependency, allow for cleaner and comparable data for the models to process, and to allow for more straightforward thresholds to be defined in the rule-based approach. The code used to create the PyTorch Datasets and DataLoaders can be found [here](dataset.py) (note: this is not in a Jupyter Notebook file as to allow for the create_dataloaders function to be imported in other files).

## Rule-Based Model

### Primary Baseline
The rule-based model relies on 3 rules to classify clips as falls:
1. Rule 1: hip descent across full clip (from average in first 10 frames to average in last 10 frames)
2. Rule 2: shoulder to hip vertical gap average in final 10 frames
3. Rule 3: hip descent in the last 10 frames (later changed to descent in any 10-frame period)
Each rule has a threshold value which needs to be surpassed for the rule to be triggered. If at least 2 of these rules are triggered in a sequence, it is classified as a fall. 

Initially, the threshold values were selected based on logical reasoning about what the normalised keypoint values represent in the context of a fall, since there was no applicable literature to follow.

For Rule 1, a value of `0.15` was chosen as a reasonable initial threshold, as it represents a 15% drop in hip position across the height of the frame, from the average position in the first 10 frames to the average position in the last 10 frames. This was chosen as a starting point on the basis that a fall from standing or a chair to the floor would be expected to produce a large downward displacement of the hips, and 15% was deemed an appropriate threshold to require before treating the descent as an indication of a fall. A 15% frame-height displacement could have an entirely different meaning depending on the camera's position, angle, and distance from the subject. For example, a higher-up camera would compress vertical motion significantly compared to a side-on view, meaning the same physical fall could produce very different absolute and normalised hip displacement values depending on the viewpoint. This sensitivity is an inherent limitation of threshold-based rules applied to either absolute or normalised image coordinates, and is one of the reasons rule-based approaches struggle to generalise across varied scenarios.

For Rule 2, a threshold of `0.10` (10% of frame height) was chosen as a reasonable initial estimate of how close together the shoulders and hips would need to be relative to the frame dimensions to indicate a collapsed or laying position, while still being selective enough to avoid triggering on more vertically collapsed viewpoints. As with Rule 1, the same issue of sensitivity to different viewpoints persists, as for example a higher-up camera would cause the shoulders and hips to appear close together even when the person is upright.

For Rule 3, a value of `0.10` was chosen as the initial threshold, representing a 10% drop in hip position over the last 10 frames (roughly the last third of a second of the clip at 30fps). The idea was that a sudden drop of that magnitude in such a short window would be a strong indicator of a fall occurring, rather than a normal movement like sitting down. Like Rules 1 and 2, what a 10% drop actually represents in the real world depends on the camera setup, and a high camera would compress vertical motion, meaning the same fall could produce a much smaller apparent drop than a side-on view would.

Once these values were selected, the model was ran on the validation data to evaluate performance. The table below shows the changes made in each iteration along with the results:

| Version | HIP_DESCENT | BODY_HORIZONTAL | RAPID_DESCENT | Rule 3 Style | Voting | Changes | F1 | Precision | Recall | Accuracy |
|---------|-------------|-----------------|---------------|--------------|--------|---------|----|-----------|--------|----------|
| v1 | 0.15 | 0.10 | 0.10 | Last 10 frames | >= 2 | Initial thresholds | 0.3883 | 0.8561 | 0.2511 | 0.6609 |
| v2 | 0.05 | 0.10 | 0.05 | Last 10 frames | >= 2 | Lowered Rules 1 & 3 thresholds | 0.6003 | 0.8290 | 0.4705 | 0.7315 |
| v3 | 0.02 | 0.10 | 0.02 | Last 10 frames | >= 2 | Lowered Rules 1 & 3 thresholds further | 0.6398 | 0.7594 | 0.5527 | 0.7333 |
| v4 | 0.02 | 0.10 | 0.02 | Last 10 frames | >= 1 | Changed voting to >= 1 | 0.7407 | 0.6289 | 0.9008 | 0.7297 |
| v5 | 0.02 | 0.10 | 0.02 | Last 10 frames | Rule 2 AND (Rule 1 OR Rule 3) | Custom voting logic | 0.5884 | 0.7852 | 0.4705 | 0.7179 |
| v6 | 0.02 | 0.20 | 0.02 | Last 10 frames | >= 1 | Raised BODY_HORIZONTAL aggressively | 0.6583 | 0.4922 | 0.9937 | 0.5579 |
| v7 | 0.02 | 0.10 | 0.12 | Full scan | >= 1 | Rule 3 redesigned to scan full clip | 0.7468 | 0.6210 | 0.9367 | 0.7278 |
| v8 | 0.02 | 0.10 | 0.02 | Full scan | >= 2 | Lowered RAPID_DESCENT, restored voting >= 2 | 0.7374 | 0.6695 | 0.8207 | 0.7495 |
| v9 | 0.05 | 0.12 | 0.02 | Full scan | >= 2 | Raised Rules 1 & 2 thresholds for differentiation | 0.7505 | 0.6757 | 0.8439 | 0.7595 |
| v10 | 0.10 | 0.12 | 0.02 | Full scan | >= 2 | Raised HIP_DESCENT further | 0.7514 | 0.6915 | 0.8228 | 0.7667 |

(Versions' individual code was not saved separately, the main code file was edited for each iteration and is available [here](rule-based.ipynb))

Key notes:
1. The `Changes` column briefly describes the changes made from the last iteration.
2. The `Voting` column shows the logic of how many rules need to be triggered in a sequence to classify it as a fall.
3. the `Rule 3 Style` column shows the change in logic of the design of Rule 3. This is because from v7 onwards, rather than checking for a rapid descent over the last 10 frames of a sequence, the rule checks for a rapid descent over any 10-frame period in the sequence, scanning all the way from the frame 0-9 window to the frame 50-59 window. It's worth noting that this change was implemented as inspiration from the alternative baseline model, which was developed after v6 to compare whether classifying individual frames in order to classify sequences would lead to better performance.

### Alternative Rule-Based Model 
The alternative rule-based model focuses on individual frame rules rather than rules across the whole sequence. Each frame is evaluated individually, and a sequence is only classified as a fall if enough frames within it are classified as fall-frames. A frame is classified as a fall-frame if at least `MIN_RULES_PER_FRAME` of the following 3 rules trigger:
1. Rule 1: hip descent of more than `HIP_VELOCITY_THRESHOLD` over a specified number of frames (`VELOCITY_WINDOW`)
2. Rule 2: shoulder to hip vertical gap is less than `BODY_HORIZONTAL_THRESHOLD`
3. Rule 3: body width-to-height ratio is more than `RATIO_THRESHOLD` (wider than it is tall)
If the number of fall-frames in a sequence meets or exceeds `FRAME_VOTE_THRESHOLD`, the sequence is classified as a fall.

Same as the original baseline, initial values were selected based on logical reasoning of appropriate starting points for each rule.

For Rule 1, a `HIP_VELOCITY_THRESHOLD` of `0.02` and a `VELOCITY_WINDOW` of `5` frames were chosen as the initial values. The velocity window of 5 frames (one sixth of a second at 30fps) was chosen as a short enough window to capture the sudden movement of a fall, without being so short that it misses those movements. The threshold of 0.02 was chosen as a starting point, representing a 2% drop in hip position over 5 frames, which would be a small but quick displacement that would be expected to occur during a rapid fall but not during normal movement.

For Rule 2, a `BODY_HORIZONTAL_THRESHOLD` of `0.10` was carried over from the primary model, where it consistently produced the most stable Rule 2 behaviour, making it a sensible starting point for the alternative model too.

For Rule 3, a `RATIO_THRESHOLD` of `1.0` was chosen as the initial value, meaning the rule triggers when the person's bounding box is at least as wide as it is tall. This is a natural starting point since a ratio of exactly 1.0 represents the boundary between a taller-than-wide and wider-than-tall skeleton, and a person who has fallen and is lying on the ground would generally be expected to have a wider-than-tall skeleton bounding box.

For the voting parameters, `MIN_RULES_PER_FRAME` was set to `2` and `FRAME_VOTE_THRESHOLD` was set to `5`. Requiring at least 2 rules to trigger per frame carried over the >= 2 voting logic used in the original baseline, and was chosen to avoid classifying a frame as a fall based on a single rule. A frame vote threshold of 5 out of 60 frames was chosen as a deliberately low starting point, with the logic that even a brief fall event should produce at least a few fall-frames, while avoiding setting it so high that short or partially captured falls would be missed.

The table below shows the changes made in each iteration along with the results:

| Version | HIP_VELOCITY | BODY_HORIZONTAL | RATIO | VELOCITY_WINDOW | MIN_RULES | FRAME_VOTE | Changes | F1 | Precision | Recall | Accuracy |
|---------|-------------|-----------------|-------|-----------------|-----------|------------|---------|----|-----------|--------|----------|
| v1 | 0.02 | 0.10 | 1.00 | 5 | 2 | 5 | Initial parameters | 0.7030 | 0.6854 | 0.7215 | 0.7387 |
| v2 | 0.01 | 0.10 | 1.00 | 5 | 2 | 10 | Lowered HIP_VELOCITY, raised FRAME_VOTE | 0.6761 | 0.6998 | 0.6540 | 0.7315 |
| v3 | 0.005 | 0.10 | 1.00 | 5 | 2 | 5 | Lowered HIP_VELOCITY further, restored FRAME_VOTE | 0.7154 | 0.6649 | 0.7743 | 0.7360 |
| v4 | 0.005 | 0.10 | 1.25 | 5 | 2 | 5 | Raised RATIO_THRESHOLD | 0.7184 | 0.6798 | 0.7616 | 0.7441 |

(Versions' individual code was not saved separately, the main code file was edited for each iteration and is available [here](alt-rule-based/alt-rule-based.ipynb))

The best result achieved was an F1 of `0.7184`, which did not surpass the original baseline's best at the time of `0.7407`, so the original baseline's structure and logic was declared to be overall better, and it was returned to and further optimised, particularly with the addition of scanning over the sequence for Rule 3 as mentioned above. The per-frame approach is included here to show that this alternative design was considered and evaluated.

### Final Version
The final version of the rule-based mode selected was `v10` of the original baseline.

## LSTM Model

The LSTM model is a 2-layer LSTM with a `hidden size` of `128`, followed by two fully connected layers `(128 -> 64 -> 2)`. The input to the model is a sequence of 17 keypoints, each with an x coordinate, y coordinate, and a confidence score, giving an `input size` of `51` per frame. The model was trained using the Adam optimiser with a ReduceLROnPlateau learning rate scheduler `(factor=0.5, patience=5)`, and the best checkpoint was saved based on validation F1.

The initial configuration was chosen as a reasonable starting point before any regularisation was introduced, to establish a clean baseline to build from:
- `DROPOUT = 0.3` was chosen as a moderate starting value, low enough that the model would still be able to learn effectively in early training, but high enough to provide some regularisation from the start
- `LEARNING_RATE = 0.001` was chosen as a starting point that is fast enough to see meaningful progress within 50 epochs, without being so large that it risks overshooting during optimisation
- `BATCH_SIZE = 32` was chosen as a balance between training stability and generalisation, as larger batches produce more stable gradient estimates but tend to not generalise as well, while smaller batches are noisier but can generalise better
- `NUM_EPOCHS = 50` was chosen as enough epochs to observe whether the model was converging, without committing to a long training run before knowing whether the architecture and configuration were working
- No class weighting, weight decay, early stopping, or gradient clipping were included in the first version, to establish a clean baseline before adding regularisation

The table below shows the changes made in each iteration along with the results:

| Version | Dropout | Weight Decay | Class Weight | LR | Epochs | Patience | Batch | Best Epoch | F1 | Precision | Recall | Accuracy |
|---------|---------|--------------|--------------|-------|--------|----------|-------|------------|----|-----------|--------|----------|
| v1 | 0.3 | 0 | — | 0.001 | 50 | — | 32 | 39 | 0.8860 | 0.8946 | 0.8776 | 0.9033 |
| v2 | 0.3 | 1e-4 | 1.5 | 0.001 | 50 | 10 | 32 | 35 | 0.8738 | 0.8639 | 0.8840 | 0.8906 |
| v3 | 0.3 | 5e-4 | 1.2 | 0.001 | 50 | 10 | 32 | 31 | 0.8722 | 0.8732 | 0.8713 | 0.8906 |
| v4 | 0.4 | 5e-4 | 1.5 | 0.001 | 50 | 10 | 32 | 45 | 0.8661 | 0.8589 | 0.8734 | 0.8843 |
| v5 | 0.4 | 5e-4 | 1.3 | 0.001 | 75 | 15 | 32 | 46 | 0.8808 | 0.8734 | 0.8882 | 0.8969 |
| v6 | 0.4 | 1e-3 | 1.3 | 0.001 | 75 | 15 | 32 | 32 | 0.8528 | 0.8219 | 0.8861 | 0.8689 |
| v7 | 0.4 | 5e-4 | 1.2 | 0.001 | 75 | 15 | 32 | 37 | 0.8628 | 0.8565 | 0.8692 | 0.8816 |
| v8 | 0.4 | 5e-4 | 1.3 | 5e-4 | 75 | 15 | 32 | 50 | 0.8580 | 0.8553 | 0.8608 | 0.8779 |
| v9 | 0.4 | 5e-4 | 1.3 | 0.001 | 75 | 15 | 64 | 54 | 0.8650 | 0.8715 | 0.8586 | 0.8852 |
| v10 | 0.4 | 5e-4 | 1.3 | 0.001 | 75 | 15 | 32 | 38 | 0.8598 | 0.8407 | 0.8797 | 0.8770 |

(Versions' individual code was not saved separately, the main code file was edited for each iteration and is available [here](LSTM-train.ipynb). However, best model checkpoints, training history, validation results, and analysis graphs were all saved for each iteration, and are all available [here](models).)

Key notes:
1. v1 through v5 were developed sequentially, with each version building on the last.
2. v6, v7, v8, v9, and v10 each branched independently from v5 to test a single isolated change, rather than continuing sequentially. This was to avoid compounding changes making it unclear which change caused which effect.
3. v10 was an ablation study removing confidence scores from the input (`input size = 34` instead of 51), to test whether the YOLO confidence scores were contributing meaningful signal. Although v10's F1 was lower than v5, the training graphs suggested the model had not yet stabilised and showed no major signs of overfitting at the point early stopping triggered, meaning further training with a higher patience or adjusted early stopping criteria could potentially have closed the gap, but time constraints did not allow for further testing. Confidence scores were nonetheless retained in the final model given v5's superior performance under the same training conditions.

To evaluate each iteration of the model and identify changes to be tested, training and validation analysis graphs were generated using [LSTM-analyse.ipynb](LSTM-analyse.ipynb). It generates loss and accuracy curves, F1/precision/recall per class and curves over epochs, and confusion matrices for each version, which were used to assess whether the model was overfitting, whether it had converged, how well it was performing, and to inform what changes to try next.

### Final Version
The final version of the LSTM selected was `v5`, which achieved an F1 of `0.8808` and the best overall balance of F1, precision, and recall on the validation set. The checkpoint for this version of the model along with it's data is available [here](models/lstm_v5) (in models) and [here](final-model) separately.

## Test Set Evaluation Results & Model Comparison

Test set evaluation and model comparison was carried out in [model-test.ipynb](model-test.ipynb). The created graphs are available [here](evaluation_graphs.png) and [here](final-model/evaluation_graphs.png).

Both final models were evaluated once on the test set after all tuning and development was complete. Neither model had been exposed to the test set at any point during development. The results are shown in the table below:

| Metric | LSTM v5 | Rule-Based v10 |
|--------|---------|----------------|
| Accuracy | 0.8863 | 0.7782 |
| Precision (fall) | 0.8541 | 0.7126 |
| Recall (fall) | 0.8840 | 0.8031 |
| F1 (fall) | 0.8688 | 0.7551 |

The confusion matrices were as follows:

| | LSTM v5 | Rule-Based v10 |
|--|---------|----------------|
| TN | 547 | 468 |
| FP | 69 | 148 |
| FN | 53 | 90 |
| TP | 404 | 367 |

The LSTM outperformed the rule-based model across all metrics on the test set. The most notable difference was in precision and false positive count, with the rule-based model producing more than twice as many false alarms (148 vs 69).
The rule call rates on the test set (Rule 1: 16.0%, Rule 2: 52.0%, Rule 3: 67.5%) were consistent with those observed on the validation set during development, which gives some confidence that the model was not overfitted to the validation data.

## References
Rahman, N.N., Mahi, A.B.S., Mistry, D., Masud, S.M.R.A., Saha, A.K., Rahman, R. and Islam, Md.R. (2025) ‘FallVision: A benchmark video dataset for fall detection’, *Data in Brief*, 59, p. 111440. Available at: [https://doi.org/10.1016/j.dib.2025.111440](https://doi.org/10.1016/j.dib.2025.111440).