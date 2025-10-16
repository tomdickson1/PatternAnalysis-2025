# Improved 3D UNet for Volumetric Segmentation of Pelvis Dataset

The files contained in this directory implement the Improved 3D UNet from [reference], and apply it to segment the pelvis dataset from [link].

## Instructions/dependencies

### Windows
To install the required dependencies, create a new conda environment and install the packages from the environment.yaml file in this directory. For example:
```bash
conda env create --name unet3d --file .\environment.yaml
```

### Others

The yaml file describes the environment setup for Windows. For other OS's, you will need to install the following packages manually:
- torch
- torchio
- numpy
- matplotlib

## Data splitting

The data was split into training, validation and testing sets, to ensure the final results on the test set would accurately reflect the model's performance on unseen samples.

Samples were allocated to each set in batches corresponding to each patient, to ensure no data leakage occured - i.e. to ensure that MRIs from the same patient did not appear in both the training and testing sets.

Some patients did not have the full number of 8 scans. The number of patients with each type of scan are as follows:
| Number of Images  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Number of Patients  | 11 | 1 | 0 | 1| 1 | 1 | 1 |  22 |

The ratio for the train, validation and test sets was set to be 8:1:2, so patients with 1 or 8 images were randomly allocates according to this ratio. The the remaining patient images remaining was 24 (2+4+5+6+7). 

According to the 8:1:2 ratio, the number of images allocate to the train, validation and test sets should be 17.45, 2.18 and 4.36. Therefore, the patient with 2 images was allocated to the validation set, the patient with 4 images was allocated to the testing set, and the remaining patients were allocated to the testing set.

The final number of images in each set was then 154, 19 and 38, which gives a ratio of 8:0.987:1.974.

## Results

Cool graph 1

Cool graph 2

Epic test performance