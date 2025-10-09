
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import time
from modules import AbstractNetwork, Improved3DUnet
from utils import get_lr


def train(network : AbstractNetwork, optimizer: optim.Optimizer, train_loader, val_loader, epochs, device=None, time_limit=0):
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
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        batches_done = 0
        epoch_start = time.time()

        total_XY = torch.zeros(1,4)
        total_X = torch.zeros(1,4)
        total_Y = torch.zeros(1,4)
        for data in tqdm(train_loader):
            data: list[torch.Tensor]
            # get the inputs; data is a list of [inputs, labels]
            x_real, x_seg = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(x_real)
            # squeeze to remove the channel dimension since its length is 1
            loss = network.loss(outputs, x_seg.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches_done += 1
        
        training_loss = running_loss / batches_done
        # scheduler.step(training_loss)
        current_lr = get_lr(optimizer)
        writer.add_scalar("Loss/train", training_loss, epoch)

        print(f'[{epoch + 1}] loss: {training_loss:.3f}, dsc: {list(2 * total_XY / (total_X + total_Y))}, lr={current_lr} ({time.time() - epoch_start:.2f} seconds)')
        if (time_limit > 0 and time.time() - start_time > time_limit):
            print(f"Time's up! Stopping at Epoch {epoch+1}")
            break

    print('Finished Training')
    writer.flush()
    torch.save(net.state_dict(), f"models/{timestamp}.model")
    torch.save(optimizer.state_dict(), f"models/{timestamp}.optim")


if __name__ == "__main__":
    pass