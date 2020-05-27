'''
The function of this file: realize the forward propagation, back propagation
of the model, loss function calculation, save model, load model function
'''

import torch
import shutil
import os


class Solver:
    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, images):
        images = images.to(self.device)
        outputs = self.model(images)
        return outputs

    def tta(self, images, seg=True):
        images = images.to(self.device)

        pred_origin = self.model(images)
        preds = torch.zeros_like(pred_origin)

        images_hflp = torch.flip(images, dims=[3])
        pred_hflip = self.model(images_hflp)

        images_vflip = torch.flip(images, dims=[2])
        pred_vflip = self.model(images_vflip)

        if seg:

            pred_hflip = torch.flip(pred_hflip, dims=[3])
            pred_vflip = torch.flip(pred_vflip, dims=[2])

        preds = preds + pred_origin + pred_hflip + pred_vflip

        pred = preds / 3.0

        return pred

    def cal_loss(self, targets, predicts, criterion):
        targets = targets.to(self.device)
        return criterion(predicts, targets)

    def backword(self, optimizer, loss):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def save_checkpoint(self, save_path, state, is_best):
        torch.save(state, save_path)
        if is_best:
            print('Saving Best Model.')
            save_best_path = save_path.replace('.pth', '_best.pth')
            shutil.copyfile(save_path, save_best_path)
    
    def load_checkpoint(self, load_path):
        if os.path.isfile(load_path):
            checkpoint = torch.load(load_path)
            self.model.module.load_state_dict(checkpoint['state_dict'])
            print('Successfully Loaded from %s' % (load_path))
            return self.model
        else:
            raise FileNotFoundError("Can not find weight file in {}".format(load_path))
