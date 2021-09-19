import sys
import os
import pts_loader
import torch

def load_data(type):
    path1 = './' + type + '_data/'
    clouds = []
    for dir in os.listdir(path1):
        path2 = path1 + dir + '/'
        for file in os.listdir(path2):
            path3 = path2 + file
            clouds.append(pts_loader.load(path3))
    return torch.tensor(clouds)

def main(argv):
    pass

if __name__ == '__main__':
    main(sys.argv)
