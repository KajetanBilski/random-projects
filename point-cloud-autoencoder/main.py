import sys
import os
import pts_loader
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
from champfer_loss import ChamferLoss
import random
import matplotlib.pyplot as plt
from models import MyAutoEncoder, MyVAutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLOUD_SIZE = 1024
BATCH_SIZE = 128

def load_data(type, folder = ''):
    path1 = './' + type + '_data/'
    clouds = []
    if folder:
        path2 = path1 + folder + '/'
        for file in os.listdir(path2):
            path3 = path2 + file
            clouds.append(pts_loader.load(path3))
    else:
        for dir in os.listdir(path1):
            path2 = path1 + dir + '/'
            for file in os.listdir(path2):
                path3 = path2 + file
                clouds.append(pts_loader.load(path3))
    return clouds

def unify_size(data, cloud_size):
    new_data = []
    for cloud in data:
        if len(cloud) < cloud_size:
            continue
        elif len(cloud) > cloud_size:
            new_data.append(random.sample(cloud, cloud_size))
        else:
            new_data.append(cloud)
    return new_data

def data_to_loader(data):
    data = torch.tensor(data)
    data = TensorDataset(data)
    print(f'Dataset size: {len(data)}')
    return DataLoader(data, BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available())

def save_net(filename, state_dict):
    if filename:
        with open(filename, 'wb') as f:
            pickle.dump(state_dict, f)
        print('Network saved to ' + filename)

# def get_chamfer_dist_loss_fn():
#     chamLoss = dist_chamfer_3D.chamfer_3DDist()
#     def chamfer_dist_loss(bath_pred, batch_true):
#         dist1, dist2, _, _ = chamLoss(bath_pred, batch_true)
#         return torch.mean(dist1) + torch.mean(dist2)
#     return chamfer_dist_loss

def validate(model: nn.Module, loss_fn, dataloader: DataLoader):
    loss = 0
    all = 0
    for batch in dataloader:
        batch = batch[0].to(device)
        batch_pred = model(batch)
        all += len(batch)
        loss += loss_fn(batch_pred, batch)
    return loss / all

def train(train_dl: DataLoader, val_dl: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn, epochs = 100, print_metrics = False, save_file = ''):
    try:
        for epoch in range(epochs):
            for batch in tqdm(train_dl):
                batch = batch[0].to(device)
                batch_pred = model(batch)
                loss = loss_fn(batch_pred, batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if print_metrics:
                with torch.no_grad():
                    train_loss = validate(model, loss_fn, train_dl)
                    val_loss = validate(model, loss_fn, val_dl)
                    print(
                        f'Epoch {epoch}: '
                        f'train loss = {train_loss:.3f}, '
                        f'validation loss = {val_loss:.3f}'
                    )
        if save_file:
            save_net(save_file, model.state_dict())
    except KeyboardInterrupt:
        if save_file:
            save_net(save_file, model.state_dict())
        raise KeyboardInterrupt

def plotPCbatch(pcArray1, pcArray2, show = True, save = False, name=None, fig_count=9 , sizex = 12, sizey=3):
    
    pc1 = pcArray1[0:fig_count]
    pc2 = pcArray2[0:fig_count]
    
    fig=plt.figure(figsize=(sizex, sizey))
    
    for i in range(fig_count*2):

        ax = fig.add_subplot(2,fig_count,i+1, projection='3d')
        
        if(i<fig_count):
            ax.scatter(pc1[i,:,0], pc1[i,:,2], pc1[i,:,1], c='b', marker='.', alpha=0.8, s=8)
        else:
            ax.scatter(pc2[i-fig_count,:,0], pc2[i-fig_count,:,2], pc2[i-fig_count,:,1], c='b', marker='.', alpha=0.8, s=8)

        ax.set_xlim3d(0.25, 0.75)
        ax.set_ylim3d(0.25, 0.75)
        ax.set_zlim3d(0.25, 0.75)
            
        plt.axis('off')
        
    plt.subplots_adjust(wspace=0, hspace=0)
        
    if(save):
        fig.savefig(name + '.png')
        plt.close(fig)
    
    if(show):
        plt.show()
    else:
        return fig

def test(folder, filename):
    model = MyVAutoEncoder(CLOUD_SIZE).to(device)
    with open(filename, 'rb') as f:
        model.load_state_dict(pickle.load(f))
    test_data = load_data('test', folder)
    test_data = unify_size(test_data, CLOUD_SIZE)
    test_dl = data_to_loader(test_data)

    for batch in test_dl:
        pred_batch = model(batch[0].to(device))
        plotPCbatch(batch[0].detach(), pred_batch.cpu().detach())
        break

    exit(0)

def main(argv):
    folder = '04379243'

    if len(argv) > 1 and argv[1] == '-t':
        test(folder, argv[2])

    train_data = load_data('train', folder)
    train_data = unify_size(train_data, CLOUD_SIZE)
    train_dl = data_to_loader(train_data)

    print('Train data loaded.')

    val_data = load_data('val', folder)
    val_data = unify_size(val_data, CLOUD_SIZE)
    val_dl = data_to_loader(val_data)

    print('Validation data loaded.')

    model = MyVAutoEncoder(CLOUD_SIZE).to(device)
    loss_fn = ChamferLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train(train_dl, val_dl, model, optimizer, loss_fn, print_metrics=True, save_file='net.p')

if __name__ == '__main__':
    main(sys.argv)
