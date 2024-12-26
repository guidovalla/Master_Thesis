# Master Thesis
Hints about the code of my master thesis

The fine-tuning was based on **[MMAction2 ](https://github.com/open-mmlab/mmaction2)) tool**, so for an explanation of the installation and its functioning please refer to [its documentation](https://mmaction2.readthedocs.io/en/latest/get_started/overview.html).
After having create a proper environment following the installation procedure (and checking that everything was compatible with the CUDA version of the GPU in use), I launched the training with: 
```console
python tools/train.py ${CONFIG} > output_logs.txt
```
where CONFIG is the path to the MMAction config file (an example can be found in this repo).
The best checkpoints resulting from the training and other useful data files (pth format) are then saved on the work_dirs folder and after the training I save them on another folder in order to use them later.

One example is for testing the model with:
```console
python tools/test.py ${CONFIG} ${CHECKPOINT} --dump _outuput.pkl > output_test_logs.txt
```
where CHECKPOINT is the path of the just moved pth file.

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









