# Master_Thesis
Hints about the code of my master thesis

The fine-tuning was based on [MMAction2 ]([http://openmmlab.com/](https://github.com/open-mmlab/mmaction2)) repository. 
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


The standard procedure to visualize the Grad-CAM heatmap with MMAction:
```console
 python tools/visualizations/vis_cam.py ${CONFIG} ${CHECKPOINT} ${VIDEO} --out-filename grad_cam_vis.mp4
```




