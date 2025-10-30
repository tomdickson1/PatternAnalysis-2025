"""
Funcitons to visualise the model's output segmentations by
generating animated gif images, and produce training metric
plots from saved tensorboard files.

@author Tom Dickson
"""

from dataset import make_dataloaders, TRAIN_IDS, VAL_IDS, TEST_IDS
from train import test
from modules import Improved3DUnet, AbstractNetwork
import argparse
import torch
import torchio as tio
from tensorboard.backend.event_processing import event_accumulator, event_multiplexer
import matplotlib.pyplot as plt
import os

def visualise(index: int, network: AbstractNetwork, loader: tio.SubjectsLoader):
    """Produce animated gifs comparing the input scan, labels and predicted
    network output for the given input index in the given SubjectsLoader.

    Args:
        index (int): index within SubjectsLoader to visualise
        network (AbstractNetwork): network whose outputs are to be visualised
        loader (tio.SubjectsLoader): SubjectsLoader containing input and labels
            data
    """
    data = loader.dataset[index]
    inputs: torch.Tensor = data["inputs"][tio.DATA].half().to(device)
    labels: torch.Tensor = data["labels"][tio.DATA].long().to(device)
    print(f"Creating GIFs for: {data["case_name"]}")
    network.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            # single images have no batch dimensions, so add it using unsqueeze
            outputs = network(inputs.unsqueeze(0))
            predicted_labels = torch.argmax(outputs, dim=1)
            tio.ScalarImage(tensor=inputs.cpu()).to_gif(2, 5, f"images/test-{index}-input.gif")
            tio.LabelMap(tensor=labels.cpu()).to_gif(2, 5, f"images/test-{index}-labels.gif")
            tio.LabelMap(tensor=predicted_labels.cpu()).to_gif(2, 5, f"images/test-{index}-predicted.gif")


def read_run(timestamp: str):
    """
    Produce plots showing training metrics for the run with the given timestamp.
    Requires tensorboard logs to be saved to the directory "runs/{timestamp}".

    Args:
        timestamp (str): _description_
    """
    run_dir = os.path.join("runs", timestamp)

    mux = event_multiplexer.EventMultiplexer()
    mux.AddRunsFromDirectory(run_dir)
    mux.Reload()

    iou_runs = [name for name, data in mux.Runs().items() if "iou/val" in data["scalars"]]
    dice_score_runs = [name for name, data in mux.Runs().items() if "dice_score/val" in data["scalars"]]

    # DICE SCORE PLOT
    plt.figure()
    for class_run in dice_score_runs:
        scalars = mux.Scalars(class_run, "dice_score/val")
        x = [e.step for e in scalars]
        y = [e.value for e in scalars]
        plt.plot(x, y, label=class_run)
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.savefig("images/validation_dice_scores.png")
    
    # IOU PLOT
    plt.figure()
    for class_run in iou_runs:
        scalars = mux.Scalars(class_run, "iou/val")
        x = [e.step for e in scalars]
        y = [e.value for e in scalars]
        plt.plot(x, y, label=class_run)
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("IOU")
    plt.savefig("images/validation_iou.png")

    # training and validation loss are stored in the "." run
    plt.figure()

    for name in ["dice_loss/train", "dice_loss/val"]:
        scalars = mux.Scalars(".", name)
        x = [e.step for e in scalars]
        y = [e.value for e in scalars]
        plt.plot(x, y, label=name)
    plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("images/loss_curve.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='predict.py',
                    description='tests and visualises a saved model',
                    )
    parser.add_argument('--model', required=True)
    parser.add_argument('--test', action='store_true',
                        help="test the model on the test set")
    parser.add_argument('--data', help="path to directory containing dataset")
    parser.add_argument('--graphs', action='store_true',
                        help="create visualations of training metrics")
    parser.add_argument('--vis',
                        help="index within testing data of input image from which to produce animated GIF segmentations")
    args = parser.parse_args()

    data_folder = "data"
    if args.data:
        data_folder = args.data

    train_loader, val_loader, test_loader = make_dataloaders(
        data_folder,"semantic_MRs","semantic_labels_only",
        TRAIN_IDS, VAL_IDS, TEST_IDS, batch_size=1
    )

    n_classes = 6
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network = Improved3DUnet(n_classes, 16, 4).to(device)
    network.load_state_dict(torch.load(f"models/{args.model}.model"))
    if args.test:
        test(network, test_loader, device)
    elif args.vis is not None:
        id_to_visualise = int(args.vis)
        visualise(id_to_visualise, network, test_loader)
    elif args.graphs:
        read_run(args.model)
