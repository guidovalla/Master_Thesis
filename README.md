# Hints about the code

For my master thesis I performed some fine-tunings based on **[MMAction2](https://github.com/open-mmlab/mmaction2) repository**, using [its documentation](https://mmaction2.readthedocs.io/en/latest/get_started/overview.html).
After having created a proper environment following the installation procedure (and checking that everything was compatible with the CUDA version of the GPU in use), I launched the fine-tunings of the pretrained models with: 
```console
python tools/train.py ${CONFIG} > output_logs.txt
```
where CONFIG is the path to the MMAction model configuration file (an example can be found in this repo).
The best checkpoints resulting from the training and other useful data files (pth format) are then saved on the work_dirs folder and after the training I saved them on another folder in order to use them later.

One example for testing the model is with:
```console
python tools/test.py ${CONFIG} ${CHECKPOINT} --dump _outuput.pkl > output_test_logs.txt
```
where CHECKPOINT is the path of the moved pth format file.

## Grad-CAMs


The standard procedure to visualize with MMAction the Grad-CAM heatmap of a video (whose path is specified in VIDEO):
```console
 python tools/visualizations/vis_cam.py ${CONFIG} ${CHECKPOINT} ${VIDEO} --out-filename grad_cam_vis.mp4
```

### Extraction heatmap values

Using the customized version of the GradCAM class (`gradcam_utils.py`) and the relative python script for the visualisation (`vis_cam.py`),  it is possible produce a txt file with all the values of the heatmap for a further analysis. I suggest to replace the original GradCAM class in `mmaction2/mmaction/utils/` and to run the new python script with:
```console
python tools/visualizations/vis_cam.py ${CONFIG} ${CHECKPOINT} ${VIDEO} --file-url ${PATH}
```
The file produced contains all the values dividing the different frame with the line "Frame N - Heatmap values:". I suggest to produce a .sh file to automatize the video Grad-CAMs collection.


## Further analyses

For the comparison with human gaze data I converted the Grad-CAM heatmaps to saliency maps summing all the frame values for each video I normalized for the number of frames. Then I compared the result with human gaze data for the same video. The human fixation maps are obtained, as it is common practice, creating a discretized matrix with ones if the corresponding pixel have been fixated by at least one human experiment participant and zeros otherwise.

#### Comparison with human data

The definition of the metrics used (AUC, CC, SIM, KL Div) is in `saliency_metrics_def.py`. I started from the code in [this repository](https://github.com/tarunsharma1/saliency_metrics) (which is still in Python 2) and I adapted it to my scope.

The expected format of the two maps is the following:
  + human fixation maps (ground truth) are map with values {0, 255}
  + the saliency maps are continous maps in the set [0,255] (first they needed a normalization using the function `normalize_map`)


















