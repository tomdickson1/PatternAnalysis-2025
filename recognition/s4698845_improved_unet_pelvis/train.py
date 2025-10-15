
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import time
from modules import AbstractNetwork, Improved3DUnet
from utils import get_lr
from dataset import make_dataloaders, TRAIN_IDS, VAL_IDS, TEST_IDS
import torchio as tio


def train(net: AbstractNetwork, optimiser: optim.Optimizer, train_loader, val_loader, epochs, device, val_check_factor=2, time_limit=0):
    """Perform training with given epochs and time limit.
    If specified, restart training from previous run.
    """
    timestamp = str(time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(f'runs/{timestamp}')
    print(net)
    start_time = time.time()
    scaler = torch.amp.GradScaler("cuda")
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        batches_done = 0
        epoch_start = time.time()

        for data in tqdm(train_loader):
            data: list[torch.Tensor]
            # get the inputs; data is a list of [inputs, labels]
            inputs: torch.Tensor = data["inputs"][tio.DATA].half().to(device)
            labels: torch.Tensor = data["labels"][tio.DATA].long().to(device)
            # zero the parameter gradients
            optimiser.zero_grad(set_to_none=True)
            # forward + backward + optimize
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = net(inputs)
                # squeeze to remove the channel dimension since its length is 1
                loss = net.loss(outputs, labels)
            # loss.backward()
            scaler.scale(loss).backward()

            # optimizer.step()
            scaler.step(optimiser)

            scaler.update()
            running_loss += loss.item()
            batches_done += 1
            del inputs, labels, loss
        
        training_loss = running_loss / batches_done
        # scheduler.step(training_loss)
        current_lr = get_lr(optimiser)
        writer.add_scalar("dice_loss/train", training_loss, epoch)
        print(f'[{epoch + 1}] loss: {training_loss:.3f}, lr={current_lr} ({time.time() - epoch_start:.2f} seconds)')

        if epoch % val_check_factor == 0:
            # run a check on the validation
            print("Validation Check!")
            test(net, val_loader, device, writer, epoch)           

        # if (time_limit > 0 and time.time() - start_time > time_limit):
        #     print(f"Time's up! Stopping at Epoch {epoch+1}")
        #     break

    print('Finished Training')
    writer.flush()
    torch.save(net.state_dict(), f"models/{timestamp}.model")
    torch.save(optimiser.state_dict(), f"models/{timestamp}.optim")

def test(net : AbstractNetwork, loader, device, writer: SummaryWriter=None, epoch=None):
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
        writer.add_scalars("dice_score/val", {f"class_{i}": x.detach().cpu().item() for i,x in enumerate(dice_score)}, epoch)
        writer.add_scalars("iou/val", {f"class_{i}": x.detach().cpu().item() for i, x in enumerate(iou)}, epoch)
        writer.add_scalars("dice_loss/val", {f"class_{i}": x.detach().cpu().item() for i, x in enumerate(iou)}, epoch)

    # switch back to training mode to enable dropout
    net.train()


def main():
    # TRAIN_IDS = {'K019'}
    # VAL_IDS = {'W029'}
    # TEST_IDS = {'S028'}
    train_loader, val_loader, test_loader = make_dataloaders("data","semantic_MRs","semantic_labels_only", TRAIN_IDS, VAL_IDS, TEST_IDS)
    n_classes = 6
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    network = Improved3DUnet(n_classes, 16, 4).to(device)
    optimiser = torch.optim.Adam(network.parameters(), lr=5e-5, eps=1e-6)
    # test(network, val_loader, device)
    train(network, optimiser, train_loader, val_loader, 24, device)

    # network.load_state_dict(torch.load("models/20251009-185232.model"))
    # test(network, train_loader)

if __name__ == "__main__":
    main()