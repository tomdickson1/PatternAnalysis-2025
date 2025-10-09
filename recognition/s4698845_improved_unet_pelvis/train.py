
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import time
from modules import AbstractNetwork, Improved3DUnet
from utils import get_lr
from dataset import make_dataloaders
import torchio as tio


def train(network : AbstractNetwork, optimiser: optim.Optimizer, train_loader, val_loader, epochs, device=None, time_limit=0):
    """Perform training with given epochs and time limit.
    If specified, restart training from previous run.
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    timestamp = str(time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(f'runs/{timestamp}')
    net = network.to(device)
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
                loss = network.loss(outputs, labels)
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
        writer.add_scalar("Loss/train", training_loss, epoch)

        print(f'[{epoch + 1}] loss: {training_loss:.3f}, lr={current_lr} ({time.time() - epoch_start:.2f} seconds)')
        if (time_limit > 0 and time.time() - start_time > time_limit):
            print(f"Time's up! Stopping at Epoch {epoch+1}")
            break

    print('Finished Training')
    writer.flush()
    torch.save(net.state_dict(), f"models/{timestamp}.model")
    torch.save(optimiser.state_dict(), f"models/{timestamp}.optim")

def test(network : AbstractNetwork, test_loader, device=None):
    net = network.to(device)
    print(net)
    batches_done = 0
    running_metric = None   
    with torch.no_grad():
        for data in tqdm(test_loader):
            data: list[torch.Tensor]
            # get the inputs; data is a list of [inputs, labels]
            x_real, x_seg = data.to(device), data[1].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = net(x_real)
                # squeeze to remove the channel dimension since its length is 1
                metric = net.metric(outputs, x_seg)
                
            print(f"Image/batch {batches_done}: {metric}")
            # loss.backward()
            if running_metric is None:
                running_metric = metric
            else:
                running_metric += metric
            batches_done += 1
            del x_real, x_seg
        print(f"Average: {running_metric / batches_done}")


def main():
    train_loader, val_loader, test_loader = make_dataloaders("data","semantic_MRs","semantic_labels_only", [3,2,1],6)
    n_classes = 6
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    network = Improved3DUnet(n_classes, 16, 4)
    optimiser = torch.optim.Adam(network.parameters(), lr=5e-5, eps=1e-6)
    train(network, optimiser, train_loader, val_loader, 50)

    # network.load_state_dict(torch.load("models/20251009-185232.model"))
    # test(network, train_loader)

if __name__ == "__main__":
    main()