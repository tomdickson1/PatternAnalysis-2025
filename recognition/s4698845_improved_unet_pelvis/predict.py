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
    data = loader.dataset[index]
    print(data)
    inputs: torch.Tensor = data["inputs"][tio.DATA].half().to(device)
    labels: torch.Tensor = data["labels"][tio.DATA].long().to(device)
    print(inputs.shape)
    print(labels.shape)
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
                    description='tests and visualisation of a saved model',
                    )
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model', required=True)
    parser.add_argument('--data')
    parser.add_argument('--graphs', action='store_true')
    parser.add_argument('--vis')
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
