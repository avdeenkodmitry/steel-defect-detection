from torch import optim
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import codecs, json
import time
import pickle
from models.model import Model
from utils.cal_dice_iou import Meter
from datasets.steel_dataset import provider
from utils.set_seed import seed_torch
from solutions.config import get_seg_config
from solutions.solver import Solver


class TrainVal():
    def __init__(self, config, fold):
        self.model_name = config.model_name
        self.model = Model(self.model_name).create_model()

        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.epoch = config.epoch
        self.fold = fold


        self.model_path = os.path.join(config.save_path, config.model_name)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        if config.resume:
            weight_path = os.path.join(self.model_path, config.resume)
            self.load_weight(weight_path)


        self.solver = Solver(self.model)


        self.criterion = torch.nn.BCEWithLogitsLoss()


        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S-%d}".format(datetime.datetime.now(), fold)
        self.writer = SummaryWriter(log_dir=os.path.join(self.model_path, TIMESTAMP))
        with codecs.open(self.model_path + '/'+ TIMESTAMP + '.json', 'w', "utf-8") as json_file:
            json.dump({k: v for k, v in config._get_kwargs()}, json_file, ensure_ascii=False)

        self.max_dice_valid = 0

        self.seed = int(time.time())
        # self.seed = 1570421136
        seed_torch(self.seed)
        with open(self.model_path + '/'+ TIMESTAMP + '.pkl','wb') as f:
            pickle.dump({'seed': self.seed}, f, -1)

    def train(self, train_loader, valid_loader):

        optimizer = optim.RAdam(self.model.module.parameters(),
                               self.lr,
                               weight_decay=self.weight_decay)

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            self.epoch+10)
        global_step = 0

        for epoch in range(self.epoch):
            epoch += 1
            epoch_loss = 0
            self.model.train(True)

            tbar = tqdm.tqdm(train_loader)
            for i, samples in enumerate(tbar):

                if len(samples) == 0:
                    continue
                images, masks = samples[0], samples[1]

                masks_predict = self.solver.forward(images)
                loss = self.solver.cal_loss(masks, masks_predict, self.criterion)
                epoch_loss += loss.item()
                self.solver.backword(optimizer, loss)

                self.writer.add_scalar('train_loss', loss.item(), global_step+i)
                params_groups_lr = str()
                for group_ind, param_group in enumerate(optimizer.param_groups):
                    params_groups_lr = params_groups_lr + 'params_group_%d' % (group_ind) + ': %.12f, ' % (param_group['lr'])
                descript = "Fold: %d, Train Loss: %.7f, lr: %s" % (self.fold, loss.item(), params_groups_lr)
                tbar.set_description(desc=descript)


            lr_scheduler.step()
            global_step += len(train_loader)

            print('Finish Epoch [%d/%d], Average Loss: %.7f' % (epoch, self.epoch, epoch_loss/len(tbar)))

            loss_valid, dice_valid, iou_valid = self.validation(valid_loader)
            if dice_valid > self.max_dice_valid: 
                is_best = True
                self.max_dice_valid = dice_valid
            else:
                is_best = False
            
            state = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'max_dice_valid': self.max_dice_valid,
            }

            self.solver.save_checkpoint(os.path.join(self.model_path, '%s_fold%d.pth' % (self.model_name, self.fold)), state, is_best)
            self.writer.add_scalar('valid_loss', loss_valid, epoch)
            self.writer.add_scalar('valid_dice', dice_valid, epoch)

    def validation(self, valid_loader):
        self.model.eval()
        meter = Meter()
        tbar = tqdm.tqdm(valid_loader)
        loss_sum = 0
        
        with torch.no_grad(): 
            for i, samples in enumerate(tbar):
                if len(samples) == 0:
                    continue
                images, masks = samples[0], samples[1]

                masks_predict = self.solver.forward(images)
                loss = self.solver.cal_loss(masks, masks_predict, self.criterion)
                loss_sum += loss.item()

                meter.update(masks, masks_predict.detach().cpu())

                descript = "Val Loss: {:.7f}".format(loss.item())
                tbar.set_description(desc=descript)
        loss_mean = loss_sum/len(tbar)

        dices, iou = meter.get_metrics()
        dice, dice_neg, dice_pos = dices
        print("IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (iou, dice, dice_neg, dice_pos))
        return loss_mean, dice, iou
    
    def load_weight(self, weight_path):

        pretrained_state_dict = torch.load(weight_path)['state_dict']
        model_state_dict = self.model.module.state_dict()
        pretrained_state_dict = {k : v for k, v in pretrained_state_dict.items() if k in model_state_dict}
        model_state_dict.update(pretrained_state_dict)
        print('Loading weight from %s' % weight_path)
        self.model.module.load_state_dict(model_state_dict)


if __name__ == "__main__":
    config = get_seg_config()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    dataloaders = provider(
        config.dataset_root, 
        os.path.join(config.dataset_root, 'train.csv'),
        mean, 
        std,
        config.batch_size, 
        config.num_workers, 
        config.n_splits, 
        mask_only=config.mask_only_flag,
        crop=config.crop,
        height=config.height,
        width=config.width
        )
    for fold_index, [train_loader, valid_loader] in enumerate(dataloaders):
        if fold_index != 1:
            continue
        train_val = TrainVal(config, fold_index)
        train_val.train(train_loader, valid_loader)
