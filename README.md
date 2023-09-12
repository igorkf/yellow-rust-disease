# yellow-rust-disease

This is the 2nd-place solution for the Beyond Visible Spectrum: AI for Agriculture 2023 challenge hosted in [Kaggle](https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-P1).


<br>


## Main ideas:
- Squeeze the 125 bands to only 3 bands using a "pre-Conv2D" layer
- Augmentations: RandomHorizontalFlip, RandomVerticalFlip, GaussianBlur
- EarlyStopping scheduler
- Freeze high level features' layers


<br>


## How to reproduce the results

1. Go to [this](https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-P1/data) link, download the data, and put in the `data` folder.

2. Create a `conda` environment from an environment file:
```
conda env create -f environment.yml
```

3. Activate the `conda` environment:
```
conda deactivate  # if "base" is already activated
conda activate crop-disease
```

4. Train the model:
```
python3 -u src/train_nn.py --bs=56 > logs/train.log
```
This will create a folder like `20230912-153034_resnet18_bs56_acc0_760714`.


5. Do predictions, passing as `dir` the folder the training script just created:
```
python3 -u src/predict_nn.py --dir=20230912-153034_resnet18_bs56_acc0_760714 > logs/pred.log
```

<br>


## Results
Stratified KFold with `k=5`
- Mean accuracy: 0.760714 
- Std. accuracy: 0.02208


<br>


## Resources
- CentOS 7
- GPU: Tesla V100 with 32GB (only used around 2GB)
- Training: ~40 minutes
- Inference: ~3 minutes


<br>


## Didn't work...oops
- Randomly choose a subset of bands (0 to 60 or 65 to 125, for example)
- Trying to learn which bands should the model focus on using "learnable weights"
