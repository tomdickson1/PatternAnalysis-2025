import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import torch
import torchio as tio

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

def find_frequencies(directory: str, max_freq=8):
    names = os.listdir(directory)
    case_items: dict[str, list] = {}
    for name in names:
        # case id is the first 4 characters, e.g. "B006"
        identifier = name[:4]
        if identifier in case_items:
            case_items[identifier].append(name)
        else:
            case_items[identifier] = [name]
    
    freq_bins = [0] * max_freq
    freq_to_ids = {}
    for identifier, names in case_items.items():
        freq = len(names)
        freq_bins[freq-1] += 1
        if freq in freq_to_ids:
            freq_to_ids[freq] += [identifier]
        else:
            freq_to_ids[freq] = [identifier]

    print(list(range(1,max_freq+1)))
    print(freq_bins)

    # for this data, we get
    # [1, 2, 3, 4, 5, 6, 7, 8]
    # [11, 1, 0, 1, 1, 1, 1, 22]
    # as the frequencies.
    # therefore, choose a 8:1:2 train-val-test split, since
    # that works nicely for 11 and 22
    # there are then 24 (2+4+5+6+7) images remaining
    # so we roughly want
    # 8/11 * 24 = 17.45 images for training
    # 1/11 * 24 = 2.18 for validation
    # 2/11 * 24 = 4.36 for testing
    # therefore, give the 4 person to test, the 2 person to validation
    # and the rest (5,6,7) go to training

    torch.manual_seed(0)
    train_ids = []
    val_ids = []
    test_ids = []
    
    ratio_target = [8/11,1/11,2/11]
    # do single image patients
    train, val, test = torch.utils.data.random_split(freq_to_ids[1], ratio_target)
    train_ids += list(train)
    val_ids += list(val)
    test_ids += list(test)
    # same for full image patients
    # multiply by 2 because there are 22
    train, val, test = torch.utils.data.random_split(freq_to_ids[max_freq], ratio_target)
    train_ids += list(train)
    val_ids += list(val)
    test_ids += list(test)

    # give patient with 2 images to validation
    val_ids += freq_to_ids[2]

    # give patient with 4 images to test
    test_ids += freq_to_ids[4]

    for f in [5,6,7]:
        train_ids += freq_to_ids[f]
    
    print("Train")
    print(len(train_ids))
    print(sum(len(case_items[x]) for x in train_ids))
    print("Val")
    print(len(val_ids))
    print(sum(len(case_items[x]) for x in val_ids))
    print("Test")
    print(len(test_ids))
    print(sum(len(case_items[x]) for x in test_ids))
    return set(train_ids), set(val_ids), set(test_ids)

    

def make_dataloaders(path: str, input_dir: str, labels_dir: str, train_ids: set,
                     val_ids: set, test_ids: set, limit=None, extension=".nii.gz"):
    input_path = os.path.join(path, input_dir)
    labels_path = os.path.join(path, labels_dir)
    if limit is None:
        limit = len(os.listdir(input_path))

    input_names = [os.path.join(input_path, x) for i, x in enumerate(sorted(os.listdir(input_path))) if i < limit and x.endswith(extension)]
    labels_names = [os.path.join(labels_path, x) for i, x in enumerate(sorted(os.listdir(labels_path))) if i < limit and x.endswith(extension)]

    train_subjects = []
    val_subjects = []
    test_subjects = []
    for i in range(len(input_names)):
        subject = tio.Subject(
            inputs=tio.ScalarImage(input_names[i]),
            labels=tio.LabelMap(labels_names[i])
        )
        identifier = os.path.basename(input_names[i])[:4]
        if identifier in train_ids:
            train_subjects.append(subject)
        elif identifier in val_ids:
            val_subjects.append(subject)
        elif identifier in test_ids:
            test_subjects.append(subject)
        else:
            print("Warning: found a case that wasn't allocated to any set")

    train_transforms = [
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.OneOf({
                tio.RandomAffine(): 0.8,
                tio.RandomElasticDeformation(): 0.2
            },
            p=0.75
        )
    ]

    test_transforms = [
        tio.RescaleIntensity(out_min_max=(0, 1))
    ]
    # use a fixed seed so each run is the same
    train = tio.SubjectsDataset(train_subjects, tio.Compose(train_transforms))
    validation = tio.SubjectsDataset(val_subjects, tio.Compose(test_transforms))
    test = tio.SubjectsDataset(test_subjects, tio.Compose(test_transforms))
    
    batch_size = 1
    num_workers = 2

    train_loader = tio.SubjectsLoader(train, batch_size=batch_size,
                    shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2 if num_workers > 0 else None, persistent_workers=True)
    val_loader = tio.SubjectsLoader(validation, batch_size=batch_size,
                    shuffle=False, num_workers=num_workers, pin_memory=False, persistent_workers=True)
    test_loader = tio.SubjectsLoader(test, batch_size=batch_size,
                    shuffle=False, num_workers=num_workers, pin_memory=False, persistent_workers=True)
    
    return train_loader, val_loader, test_loader

# hardcode these value for reproduceability between different machines
TRAIN_IDS = {'K019', 'M013', 'N010', 'R016', 'K042', 'M023', 'K008', 'R024', 'H017', 'V027', 'T014', 'O025', 'H007', 'M015', 'B037', 'R039', 'B040', 'K018', 'M004', 'J026', 'D031', 'S022', 'W041', 'J005', 'M030', 'S035', 'B038'}
VAL_IDS = {'W029', 'T009', 'B006', 'S033'}
TEST_IDS = {'S028', 'G021', 'M036', 'L011', 'W012', 'M020', 'C032'}

if __name__ == "__main__":
    # testing code
    path = r"data\semantic_labels_only"
    train_ids, val_ids, test_ids = find_frequencies(path)
    # these 'should' be the same as TRAIN_IDS, VAL_IDS, TEST_IDS, except
    # for differences in RNG between machines
    train_loader, val_loader, test_loader = make_dataloaders("data","semantic_MRs","semantic_labels_only", train_ids, val_ids, test_ids)
    for number, data in enumerate(train_loader):
        image: tio.Image = data["inputs"][tio.DATA].squeeze(1)
        labels: tio.Image = data["labels"][tio.DATA].squeeze(1)
        tio.ScalarImage(tensor=image).to_gif(2, 5, f"images/{number}-input.gif")
        tio.LabelMap(tensor=labels).to_gif(2, 5, f"images/{number}-labels.gif")
        print(labels.data.unique())
        exit()
    # names = [os.path.join(path, x) for x in os.listdir(path)]
    # freqs = {}
    # for name in os.listdir(path):
    #     case = int(name[5:8])
    #     if case in freqs:
    #         freqs[case] += 1
    #     else:
    #         freqs[case] = 1
    # x = []
    # y = []
    # for case, freq in freqs.items():
    #     x.append(case)
    #     y.append(freq)
    # plt.bar(x,y)
    # plt.xlabel("Case ID")
    # plt.ylabel("Number of datapoints")
    # plt.show()
    # res = np.expand_dims(load_data_3D(names, early_stop=False, dtype=np.uint8),1)
    # print(res.shape)
    # print(np.max(res))