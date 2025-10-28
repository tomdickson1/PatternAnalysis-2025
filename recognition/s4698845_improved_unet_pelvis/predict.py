from dataset import make_dataloaders, TRAIN_IDS, VAL_IDS, TEST_IDS
from train import test
from modules import Improved3DUnet, AbstractNetwork
import argparse
import torch
import torchio as tio

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='predict.py',
                    description='tests and visualisation of a saved model',
                    )
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--model', required=True)
    parser.add_argument('--data')
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
