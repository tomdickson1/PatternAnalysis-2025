import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import torch

def to_channels(arr: np.ndarray, dtype = np.uint8) -> np.ndarray :
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype = dtype )
    for c in channels :
        c = int(c)
        res[...,c:c+1][arr == c]= 1

    return res

def load_data_3D(imageNames, normImage = False, categorical = False, dtype = np.float32 ,
    getAffines = False, orient = False, early_stop = False):
    '''
    Load medical image data from names, cases list provided into a list for each.
    This function pre - allocates 5 D arrays for conv3d to avoid excessive memory
    usage.

    normImage: bool(normalise the image 0.0 -1.0)
    orient: Apply orientation and resample image? Good for images with large slice
    thickness or anisotropic resolution
    dtype: Type of the data.If dtype = np.uint8, it is assumed that the data is
    labels
    early_stop: Stop loading pre - maturely? Leaves arrays mostly empty, for quick
    loading and testing scripts.
    '''
    affines =[]
    # ~ interp ='continuous'
    interp ='linear'
    if dtype == np.uint8: # assume labels
        interp ='nearest'

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    # TODO: ask about this im.applyOrientation thing?
    if orient:
        niftiImage = niftiImage.applyOrientation(niftiImage, interpolation = interp, scale =1)
    # ~ testResultName = " oriented.nii.gz "
    # ~ niftiImage.to_filename(testResultName )
    first_case = niftiImage.get_fdata(caching = 'unchanged')
    if len(first_case.shape)== 4:
        first_case = first_case[:,:,:,0]# sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype = dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros (( num, rows, cols, depth, channels ), dtype = dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype = dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if orient:
            niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        inImage = niftiImage.get_fdata(caching ='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape)== 4:
            inImage = inImage[: ,: ,: ,0] # sometimes extra dims in HipMRI_study data
        inImage = inImage[: ,: ,: depth] # clip slices
        inImage = inImage.astype(dtype)
        if normImage :
            # ~ inImage = inImage / np.linalg.norm(inImage )
            # ~ inImage = 255. * inImage / inImage.max ()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical :
            inImage = to_channels(inImage, dtype = dtype)
            # ~ images[i ,: ,: ,: ,:]= inImage
            images[i ,: inImage.shape[0],: inImage.shape[1],: inImage.shape[2],: inImage.shape[3]]= inImage # with pad
        else:
            # ~ images[i ,: ,: ,:]= inImage
            images[i ,: inImage.shape[0],: inImage.shape[1],: inImage.shape[2]]= inImage # with pad

        affines.append(affine)
        if i > 20 and early_stop :
            break

    if getAffines:
        return images, affines
    else:
        return images

def make_dataloaders(path: str, input_dir: str, labels_dir: str, splits: list[int], limit=None):
    input_path = os.path.join(path, input_dir)
    labels_path = os.path.join(path, labels_dir)
    if limit is None:
        limit = len(os.listdir(input_path))

    input_names = [os.path.join(input_path, x) for i, x in enumerate(os.listdir(input_path)) if i < limit]
    labels_names = [os.path.join(labels_path, x) for i, x in enumerate(os.listdir(labels_path)) if i < limit]

    # unsqueeze to add a dimension for channels
    inputs = torch.from_numpy(load_data_3D(input_names, normImage=True)).unsqueeze(1)
    
    labels = torch.from_numpy(load_data_3D(labels_names, dtype=np.uint8)).unsqueeze(1).long()

    # use a fixed seed so each run is the same
    generator = torch.Generator().manual_seed(42)
    all_data = torch.utils.data.TensorDataset(inputs, labels)
    print("Input Shape", inputs.shape)
    print("Label shape", labels.shape)

    train, validation, test = torch.utils.data.random_split(all_data, splits, generator)
    batch_size = 1
    num_workers = 0

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                    shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size,
                    shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                    shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # testing code
    path = r"Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-\data\HipMRI_study_complete_release_v1\semantic_labels_anon"
    path = r"data\semantic_labels_only"
    names = [os.path.join(path, x) for x in os.listdir(path)]
    freqs = {}
    for name in os.listdir(path):
        case = int(name[5:8])
        if case in freqs:
            freqs[case] += 1
        else:
            freqs[case] = 1
    x = []
    y = []
    for case, freq in freqs.items():
        x.append(case)
        y.append(freq)
    plt.bar(x,y)
    plt.xlabel("Case ID")
    plt.ylabel("Number of datapoints")
    plt.show()
    # res = np.expand_dims(load_data_3D(names, early_stop=False, dtype=np.uint8),1)
    # print(res.shape)
    # print(np.max(res))