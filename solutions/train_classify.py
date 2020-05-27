from torch import optim
import torch
import tqdm
from solutions.config import get_classify_config
from solutions.solver import Solver
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import codecs, json
import time

from models.model import ClassifyResNet
from utils.loss import ClassifyLoss
from datasets.steel_dataset import classify_provider
from utils.cal_classify_accuracy import Meter
from utils.set_seed import seed_torch
import pickle



class TrainVal():
    def __init__(self, config, fold):

        self.model_name = config.model_name
        self.model = ClassifyResNet(self.model_name, 4, training=True)
        if torch.cuda.is_available():
            self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.cuda()


        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.epoch = config.epoch
        self.fold = fold


        self.solver = Solver(self.model)

        self.criterion = ClassifyLoss()

        self.model_path = os.path.join(config.save_path, config.model_name)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        #
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S-%d}-classify".format(datetime.datetime.now(), fold)
        self.writer = SummaryWriter(log_dir=os.path.join(self.model_path, TIMESTAMP))
        with codecs.open(self.model_path + '/'+ TIMESTAMP + '.json', 'w', "utf-8") as json_file:
            json.dump({k: v for k, v in config._get_kwargs()}, json_file, ensure_ascii=False)

        self.max_accuracy_valid = 0
        self.seed = int(time.time())
        # self.seed = 1570421136
        seed_torch(self.seed)
        with open(self.model_path + '/'+ TIMESTAMP + '.pkl','wb') as f:
            pickle.dump({'seed': self.seed}, f, -1)

    def train(self, train_loader, valid_loader):

        optimizer = optim.Adam(self.model.module.parameters(), self.lr, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epoch+10)
        global_step = 0

        for epoch in range(self.epoch):
            epoch += 1
            epoch_loss = 0
            self.model.train(True)

            tbar = tqdm.tqdm(train_loader)
            for i, (images, labels) in enumerate(tbar):

                labels_predict = self.solver.forward(images)
                loss = self.solver.cal_loss(labels, labels_predict, self.criterion)
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

            class_neg_accuracy, class_pos_accuracy, class_accuracy, neg_accuracy, pos_accuracy, accuracy, loss_valid = \
                self.validation(valid_loader)

            if accuracy > self.max_accuracy_valid: 
                is_best = True
                self.max_accuracy_valid = accuracy
            else:
                is_best = False
            
            state = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'max_accuracy_valid': self.max_accuracy_valid,
            }

            self.solver.save_checkpoint(os.path.join(self.model_path, '%s_classify_fold%d.pth' % (self.model_name, self.fold)), state, is_best)
            self.writer.add_scalar('valid_loss', loss_valid, epoch)
            self.writer.add_scalar('valid_accuracy', accuracy, epoch)
            self.writer.add_scalar('valid_class_0_accuracy', class_accuracy[0], epoch)
            self.writer.add_scalar('valid_class_1_accuracy', class_accuracy[1], epoch)
            self.writer.add_scalar('valid_class_2_accuracy', class_accuracy[2], epoch)
            self.writer.add_scalar('valid_class_3_accuracy', class_accuracy[3], epoch)

    def validation(self, valid_loader):

        self.model.eval()
        meter = Meter()
        tbar = tqdm.tqdm(valid_loader)
        loss_sum = 0
        
        with torch.no_grad(): 
            for i, (images, labels) in enumerate(tbar):

                labels_predict = self.solver.forward(images)
                loss = self.solver.cal_loss(labels, labels_predict, self.criterion)
                loss_sum += loss.item()

                meter.update(labels, labels_predict.cpu())

                descript = "Val Loss: {:.7f}".format(loss.item())
                tbar.set_description(desc=descript)
        loss_mean = loss_sum / len(tbar)

        class_neg_accuracy, class_pos_accuracy, class_accuracy, neg_accuracy, pos_accuracy, accuracy = meter.get_metrics()
        print("Class_0_accuracy: %0.4f | Class_1_accuracy: %0.4f | Class_2_accuracy: %0.4f | Class_3_accuracy: %0.4f | "
              "Negative accuracy: %0.4f | positive accuracy: %0.4f | accuracy: %0.4f" %
              (class_accuracy[0], class_accuracy[1], class_accuracy[2], class_accuracy[3],
               neg_accuracy, pos_accuracy, accuracy))
        return class_neg_accuracy, class_pos_accuracy, class_accuracy, \
               neg_accuracy, pos_accuracy, accuracy, loss_mean


if __name__ == "__main__":
    config = get_classify_config()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    dataloaders = classify_provider(
        config.dataset_root, 
        os.path.join(config.dataset_root, 'train.csv'), 
        mean, 
        std, 
        config.batch_size, 
        config.num_workers, 
        config.n_splits,
        crop=config.crop,
        height=config.height,
        width=config.width
        )
    for fold_index, [train_loader, valid_loader] in enumerate(dataloaders):
        if fold_index != 1:
            continue
        train_val = TrainVal(config, fold_index)
        train_val.train(train_loader, valid_loader)

