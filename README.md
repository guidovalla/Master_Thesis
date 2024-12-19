# Master_Thesis
Hints about the code of my master thesis

The fine-tuning was based on [MMAction2 ]([http://openmmlab.com/](https://github.com/open-mmlab/mmaction2)) repository. 
After having create a proper environment following the installation procedure (and checking that everything was compatible with the CUDA version of the GPU in use), I launched the training with: 
```console
python tools/train.py ${CONFIG} > output_logs.txt
```
where CONFIG is path to the MMAction config file (an example of which can be found in this repo).
The best checkpoints resulting from the training and other useful data file are then saved on the work_dirs folder and after the training I save them on another folder in order to use them later.




