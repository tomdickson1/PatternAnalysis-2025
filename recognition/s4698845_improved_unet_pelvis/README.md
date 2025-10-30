# Improved 3D UNet for Volumetric Segmentation of Pelvis Dataset

The files contained in this directory implement the Improved 3D UNet from [reference], and apply it to segment the pelvis dataset from [link].

| Input | Ground Truth | Generated Segmentation |
| :---: | :---: | :---: |
| ![](images/test-0-input.gif)  | ![](images/test-0-labels.gif) | ![](images/test-0-predicted.gif) |


## Instructions/dependencies

### Windows with conda
To install the required dependencies, create a new conda environment and install the packages from the environment.yaml file in this directory. For example:
```bash
conda env create --name unet3d --file .\environment.yaml
```

### Others with pip

The yaml file describes the environment setup for Windows using conda. For other OS's, you will need to install the following packages manually:
- torch (2.9.0+cu126)
- torchio (0.20.23)
- numpy (2.3.3)
- matplotlib (3.10.7)
- tensorboard (2.20.0)
- tqdm (4.67.1)

Alternatively, the `requirements.txt` file may be installed by activating a python venv and running
```bash
pip install -r requirments.txt
```


## Data splitting

The data was split into training, validation and testing sets, to ensure the final results on the test set would accurately reflect the model's performance on unseen samples.

Samples were allocated to each set in batches corresponding to each patient, to ensure no data leakage occured - i.e. to ensure that MRIs from the same patient did not appear in both the training and testing sets.

Some patients did not have the full number of 8 scans. The number of patients with each type of scan are as follows:
| Number of Images  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Number of Patients  | 11 | 1 | 0 | 1| 1 | 1 | 1 |  22 |

The ratio for the train, validation and test sets was set to be 8:1:2, so patients with 1 or 8 images were randomly allocates according to this ratio. The the remaining patient images remaining was 24 (2+4+5+6+7). 

According to the 8:1:2 ratio, the number of images allocate to the train, validation and test sets should be 17.45, 2.18 and 4.36. Therefore, the patient with 2 images was allocated to the validation set, the patient with 4 images was allocated to the testing set, and the remaining patients were allocated to the testing set.

The final number of images in each set was then 154, 19 and 38, which gives a ratio of 8:0.987:1.974 - close to the target split ratio.

## Network Structure

### Layers

[image of network from paper]

explanation of key bits?

### Loss Function

Soft dice loss

### Optimisation

Half precision

## Results

The following table gives the test set dice scores for each class, showing that all were above the 0.8 threshold.

| Class | Background (0) | Body (1) | Bones (2) | Bladder (3) | Rectum (4) | Prostate (5) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dice Score on Test Set**  | 0.996 | 0.961 | 0.901 | 0.947 | 0.836 | 0.844 |

The following figure shows the behaviour as the model trained by plotting the dice loss under both the training
and validation sets. The validation loss can be seen to plateau while the training curve still slightly decreases, indicating that further training of the model would result in overfitting.

![](images/loss_curve.png)

The following plot shows the improvement in validation set dice scores as the training progressed, showing how all reached above the 0.80 threshold by the final epoch. Note that the dice scores in the above table are for the *test* set, not the validation set. The validation set was montiored during training, while the test set was only used once at the end to assess the performance of the final model.

![](images/validation_dice_scores.png)