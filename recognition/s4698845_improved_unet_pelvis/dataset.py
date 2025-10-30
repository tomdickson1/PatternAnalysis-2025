"""
Functions to analyse data frequencies (how many images per human case)
and perform splitting into training, validation and test sets.
Data splits are hardcoded for reproduceability between machines, but the code
that generated the splits is provided in `split_data()`.

@author Tom Dickson
"""

import os
import torch
import torchio as tio

def split_data(directory: str, max_freq=8):
    """Allocate case ids to training, validation and test
    sets, targetting a ratio of 8:1:2.

    Args:
        directory (str): directory containing case files (may be input
            images or labels)
        max_freq (int, optional): highest number of images possible per case
            id (human subject). Defaults to 8.

    Returns:
        tuple[set, set, set]: sets of ids for the training, validation and test
            sets
    """
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
                     val_ids: set, test_ids: set, limit: int = None,
                     extension=".nii.gz", batch_size=1
                     ) -> tuple[tio.SubjectsLoader, tio.SubjectsLoader, tio.SubjectsLoader]:
    """Create Torchio Dataloaders from images and labels under the given path,
    split according to the given sets of ids for training, validation and test sets.
    A limit on the number of images to import, the file extension to search for,
    and the batch size of the created dataloaders can also be specified.

    Args:
        path (str): path of parent directory of the input image directory and
            label image directory.
        input_dir (str): name of input image directory
        labels_dir (str): name of labels directory
        train_ids (set): case ids corresponding to the training set
        val_ids (set): case ids corresponding to the validation set
        test_ids (set): case ids corresponding to the test set
        limit (int, optional): Max images to import. Defaults to None.
        extension (str, optional): File extension of images and labels. Defaults to ".nii.gz".
        batch_size (int, optional): Batch size of created dataloaders. Defaults to 1.

    Returns:
        Tuple[tio.SubjectsLoader, tio.SubjectsLoader, tio.SubjectsLoader]: train, validation and test SubjectsLoaders
    """
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

    # apply data augmentation only to the training set
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
    train_ids, val_ids, test_ids = split_data(path)
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