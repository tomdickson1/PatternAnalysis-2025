"""
Functions to facilitate training, validating and saving the 3D Improved UNet
model.

@author Tom Dickson
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import time
from modules import AbstractNetwork, Improved3DUnet
from utils import get_lr
from dataset import make_dataloaders, TRAIN_IDS, VAL_IDS, TEST_IDS
import torchio as tio
import argparse

# number of classes in the dataset
N_CLASSES = 6
INITIAL_UNET_CHANNELS = 16
UNET_DEPTH = 4

def train(net: AbstractNetwork, optimiser: optim.Optimizer,
          train_loader: tio.SubjectsLoader, val_loader: tio.SubjectsLoader,
          epochs: int, device: str, val_check_factor: int =2):
    """Train the given network with the given optimiser, using data from the 
    given training loader. 

    Args:
        net (AbstractNetwork): network to train
        optimiser (optim.Optimizer): optimiser to use
        train_loader (tio.SubjectsLoader): dataloader for training data
        val_loader (tio.SubjectsLoader): dataloader for validation data
        epochs (int): number of epochs to train for
        device (str): pytorch device to use
        val_check_factor (int, optional): calculate validation statistics only
            every `val_check_factor` epochs. Useful on slow machines. Defaults to 2.
    """
    timestamp = str(time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(f'runs/{timestamp}')
    print(net)
    print(f"Started training at {timestamp}")
    scaler = torch.amp.GradScaler("cuda")
    
    for epoch in range(epochs):
        running_loss = 0.0
        batches_done = 0
        epoch_start = time.time()

        for data in tqdm(train_loader):
            data: list[torch.Tensor]
            inputs: torch.Tensor = data["inputs"][tio.DATA].half().to(device)
            labels: torch.Tensor = data["labels"][tio.DATA].long().to(device)
            # zero the parameter gradients
            optimiser.zero_grad(set_to_none=True)
            # forward + backward + optimize
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = net(inputs)
                # squeeze to remove the channel dimension since its length is 1
                loss = net.loss(outputs, labels)

            # use the gradient scaler to account for differences due to autocasting
            scaler.scale(loss).backward()
            scaler.step(optimiser)

            scaler.update()
            running_loss += loss.item()
            batches_done += 1
            del inputs, labels, loss, outputs
        
        training_loss = running_loss / batches_done
        current_lr = get_lr(optimiser)
        writer.add_scalar("dice_loss/train", training_loss, epoch)
        print(f'[{epoch + 1}] loss: {training_loss:.3f}, lr={current_lr} ({time.time() - epoch_start:.2f} seconds)')

        if epoch % val_check_factor == 0:
            # run a check on the validation
            print("Validation Check!")
            test(net, val_loader, device, writer, epoch)

    print('Finished Training')
    writer.flush()
    torch.save(net.state_dict(), f"models/{timestamp}.model")
    torch.save(optimiser.state_dict(), f"models/{timestamp}.optim")

def test(net : AbstractNetwork, loader: tio.SubjectsLoader, device: str,
         writer: SummaryWriter=None, epoch=None):
    """Test the given network with data from the given SubjectsLoader. If using
    inside a training loop, can supply a tensorboard summary writer and epoch
    number so data can be saved.

    Args:
        net (AbstractNetwork): network to test
        loader (tio.SubjectsLoader): source of testing data
        device (str): pytorch device to use
        writer (SummaryWriter, optional): Tensorboard writer to use for saving data.
            If not provided, data is not saved
        epoch (int, optional): epoch number to use when writing to Tensorboard. Defaults to None.
    """
    net.eval()
    with torch.no_grad():
        running_xy = torch.zeros((1, net.n_classes), device=device)
        running_x = torch.zeros((1, net.n_classes), device=device)
        running_y = torch.zeros((1, net.n_classes), device=device)
        running_loss = 0.0
        for data in tqdm(loader):
            data: list[torch.Tensor]
            inputs: torch.Tensor = data["inputs"][tio.DATA].half().to(device)
            labels: torch.Tensor = data["labels"][tio.DATA].long().to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = net(inputs)
                xy, x, y = net.metric(outputs, labels)
                running_loss += net.loss(outputs, labels).item()
            running_x += x 
            running_y += y 
            running_xy += xy 
            del inputs, labels, xy, x, y
        dice_score = (2 * running_xy / (running_x + running_y)).flatten()
        iou = dice_score / (2 - dice_score)
    print(f"Dice Scores: {dice_score.detach().cpu().tolist()}")
    print(f"IOU: {iou.detach().cpu().tolist()}")
    if writer:
        writer.add_scalars("dice_score/val",
            {f"class_{i}": x.detach().cpu().item() for i,x in enumerate(dice_score)},
            epoch)
        writer.add_scalars("iou/val",
            {f"class_{i}": x.detach().cpu().item() for i, x in enumerate(iou)},
            epoch)
        writer.add_scalar("dice_loss/val", running_loss / len(loader), epoch)

    # switch back to training mode to enable dropout/training specific operations
    net.train()


def main():
    """Function to execute functions according to user
    specified arguments.
    """
    parser = argparse.ArgumentParser(
                    prog='train.py',
                    description='trains the 3D Improved UNet Model',
                    )
    parser.add_argument('--test',
                        action='store_true',
                        help="test the model on the test set (requires specifying --prev)")
    parser.add_argument('--compile',
                        action='store_true',
                        help="JIT compile the model for faster performance (Linux only)")
    parser.add_argument('--prev',
                        help="timestamp of previous model to load, YYYYMMDD-HHMMSS.If not provided, train a new model")
    parser.add_argument('--epochs', help="number of epochs to train for")
    parser.add_argument('--data', help="path to directory containing dataset")
    parser.add_argument('--batch', default=1,
                        help="batch size (only 1 is supported currently due to data augmentation)")
    args = parser.parse_args()
    data_folder = "data"
    if args.data:
        data_folder = args.data
    train_loader, val_loader, test_loader = make_dataloaders(
        data_folder,"semantic_MRs","semantic_labels_only",
        TRAIN_IDS, VAL_IDS, TEST_IDS, batch_size=int(args.batch)
        )
        

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network = Improved3DUnet(N_CLASSES, INITIAL_UNET_CHANNELS, UNET_DEPTH).to(device)
    if args.compile:
        net_to_use = torch.compile(network)
    else:
        net_to_use = network
    optimiser = torch.optim.Adam(net_to_use.parameters(), lr=5e-5, eps=1e-6)
    if args.prev:
        net_to_use.load_state_dict(torch.load(f"models/{args.prev}.model"))
        optimiser.load_state_dict(torch.load(f"models/{args.prev}.optim"))
    if args.test:
        test(net_to_use, test_loader, device)
    else:
        try:
            epochs = int(args.epochs)
        except:
            print("Invalid epochs specification!")
            exit()
        train(net_to_use, optimiser, train_loader, val_loader, epochs, device, val_check_factor=1)

if __name__ == "__main__":
    main()
