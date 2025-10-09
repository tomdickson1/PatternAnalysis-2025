
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import time
from modules import AbstractNetwork, Improved3DUnet
from utils import get_lr
from dataset import make_dataloaders


def train(network : AbstractNetwork, optimizer: optim.Optimizer, train_loader, val_loader, epochs, device=None, time_limit=0):
    """Perform training with given epochs and time limit.
    If specified, restart training from previous run.
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    timestamp = str(time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(f'runs/{timestamp}')
    net = network.to(device).half()
    print(net)
    start_time = time.time()
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        batches_done = 0
        epoch_start = time.time()

        for data in tqdm(train_loader):
            data: list[torch.Tensor]
            # get the inputs; data is a list of [inputs, labels]
            x_real, x_seg = data[0].to(device, dtype=torch.float16), data[1].to(device)
            # zero the parameter gradients
            print("Loaded data")
            optimizer.zero_grad()

            # forward + backward + optimize
            print("Input shape ", x_real.shape)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = net(x_real)
                # squeeze to remove the channel dimension since its length is 1
                loss = network.loss(outputs, x_seg)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches_done += 1
            del x_real, x_seg, loss
        
        training_loss = running_loss / batches_done
        # scheduler.step(training_loss)
        current_lr = get_lr(optimizer)
        writer.add_scalar("Loss/train", training_loss, epoch)

        print(f'[{epoch + 1}] loss: {training_loss:.3f}, lr={current_lr} ({time.time() - epoch_start:.2f} seconds)')
        if (time_limit > 0 and time.time() - start_time > time_limit):
            print(f"Time's up! Stopping at Epoch {epoch+1}")
            break

    print('Finished Training')
    writer.flush()
    torch.save(net.state_dict(), f"models/{timestamp}.model")
    torch.save(optimizer.state_dict(), f"models/{timestamp}.optim")


if __name__ == "__main__":
    train_loader, val_loader, test_loader = make_dataloaders("data","semantic_MRs","semantic_labels_only", [2,1,1],4)
    n_classes = 6
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    network = Improved3DUnet(n_classes, 16, 4).half()
    network.compile()
    optimiser = torch.optim.Adam(network.parameters(), lr=1e-3)
    train(network, optimiser, train_loader, val_loader, 5)