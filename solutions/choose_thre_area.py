import os
import cv2
import tqdm
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import codecs
import json
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from solutions.solver import Solver
from models.model import Model
from datasets.steel_dataset import provider
from utils.cal_dice_iou import compute_dice_class
from solutions.config import get_seg_config


class ChooseThresholdMinArea:
    '''
    Choose pixel threshold and minimum connected domain for each category
    '''
    def __init__(self, model, model_name, valid_loader, fold, save_path, class_num=4):
        '''
        Model initialization
        Args:
            model: Used model
            model_name: The name of the current model
            valid_loader: Dataloader with data verifing
            save_path: Path to save results
            class_num: quantity of classes
        '''
        self.model = model
        self.model_name = model_name
        self.valid_loader = valid_loader
        self.fold = fold
        self.save_path = save_path
        self.class_num = class_num

        self.model.eval()
        self.solver = Solver(model)

    def choose_threshold_minarea(self):
        '''
        Use the grid method to search the optimal pixel threshold and the
        optimal minimum connected domain of each category, and draw the heat
        map during the search of each category
        Return:
            best_thresholds_little: Optimal threshold for each class
            best_minareas_little: Optimal minimum connected residue
            of each class
            max_dices_little: Maximum dice value for each class
        '''
        init_thresholds_range, init_minarea_range = np.arange(0.50, 0.71, 0.03), np.arange(768, 2305, 256)


        thresholds_table_big = np.array([init_thresholds_range, init_thresholds_range, \
                                         init_thresholds_range, init_thresholds_range])
        minareas_table_big = np.array([init_minarea_range, init_minarea_range, \
                                       init_minarea_range, init_minarea_range])

        f, axes = plt.subplots(figsize=(28.8, 18.4), nrows=2, ncols=self.class_num)
        cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)

        best_thresholds_big, best_minareas_big, max_dices_big = self.grid_search(thresholds_table_big, minareas_table_big, axes[0,:], cmap)
        print('best_thresholds_big:{}, best_minareas_big:{}, max_dices_big:{}'.format(best_thresholds_big, best_minareas_big, max_dices_big))


        thresholds_table_little, minareas_table_little = list(), list()

        for best_threshold_big, best_minarea_big in zip(best_thresholds_big, best_minareas_big):
            thresholds_table_little.append(np.arange(best_threshold_big-0.03, best_threshold_big+0.03, 0.015))
            minareas_table_little.append(np.arange(best_minarea_big-256, best_minarea_big+257, 128))
        thresholds_table_little, minareas_table_little = np.array(thresholds_table_little), np.array(minareas_table_little)

        best_thresholds_little, best_minareas_little, max_dices_little = self.grid_search(thresholds_table_little, minareas_table_little, axes[1,:], cmap)
        print('best_thresholds_little:{}, best_minareas_little:{}, max_dices_little:{}'.format(best_thresholds_little, best_minareas_little, max_dices_little))

        f.savefig(os.path.join(self.save_path, self.model_name + '_fold'+str(self.fold)), bbox_inches='tight')

        plt.close()

        return best_thresholds_little, [float(x) for x in best_minareas_little], max_dices_little

    def grid_search(self, thresholds_table, minareas_table, axes, cmap):
        '''
        Given thresholds_table and minareas_table containing the search
        interval of each category, find the optimal pixel threshold of each
        category, the optimal minimum connected domain, and the highest dice
        And draw the heat map of the search process of each category

        Args:
            thresholds_table: threshold range to be searched, dimension is
            [4, N], numpy type
            minareas_table: minimum connected domain range to be searched,
            dimension is [4, N], numpy type
            axes: the handle needed to draw the heat map for each category,
            the size is [class_num]
            cmap: cmap required when drawing

        '''
        dices_table = np.zeros((self.class_num, np.shape(thresholds_table)[1], np.shape(minareas_table)[1]))
        tbar = tqdm.tqdm(self.valid_loader)
        with torch.no_grad():
            for i, samples in enumerate(tbar):
                if len(samples) == 0:
                    continue
                images, masks = samples[0], samples[1]

                masks_predict_allclasses = self.solver.forward(images)
                dices_table += self.grid_search_batch(thresholds_table, minareas_table, masks_predict_allclasses, masks)

        dices_table = dices_table/len(tbar)
        best_thresholds, best_minareas, max_dices = list(), list(), list()

        for each_class, dices_oneclass_table in enumerate(dices_table):
            max_dice = np.max(dices_oneclass_table)
            max_location = np.unravel_index(np.argmax(dices_oneclass_table, axis=None),
                                            dices_oneclass_table.shape)
            best_thresholds.append(thresholds_table[each_class, max_location[0]])
            best_minareas.append(minareas_table[each_class, max_location[1]])
            max_dices.append(max_dice)

            data = pd.DataFrame(data=dices_oneclass_table, index=np.around(thresholds_table[each_class,:], 3), columns=minareas_table[each_class,:])
            sns.heatmap(data, linewidths=0.05, ax=axes[each_class], vmax=np.max(dices_oneclass_table), vmin=np.min(dices_oneclass_table), cmap=cmap,
                        annot=True, fmt='.4f')
            axes[each_class].set_title('search result')
        return best_thresholds, best_minareas, max_dices

    def grid_search_batch(self, thresholds_table, minareas_table, masks_predict_allclasses, masks_allclasses):


        dices_table = list()
        for each_class, (thresholds_range, minareas_range) in enumerate(zip(thresholds_table, minareas_table)):

            masks_predict_oneclass = masks_predict_allclasses[:, each_class, ...]
            masks_oneclasses = masks_allclasses[:, each_class, ...]

            dices_range = self.post_process(thresholds_range, minareas_range, masks_predict_oneclass, masks_oneclasses)
            dices_table.append(dices_range)

        return np.array(dices_table)

    def post_process(self, thresholds_range, minareas_range, masks_predict_oneclass, masks_oneclasses):
        masks_predict_oneclass = torch.sigmoid(masks_predict_oneclass).detach().cpu().numpy()
        dices_range = np.zeros((len(thresholds_range), len(minareas_range)))

        for index_threshold, threshold in enumerate(thresholds_range):
            for index_minarea, minarea in enumerate(minareas_range):
                batch_preds = list()

                for pred in masks_predict_oneclass:
                    mask = cv2.threshold(pred, threshold, 1, cv2.THRESH_BINARY)[1]

                    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
                    predictions = np.zeros((256, 1600), np.float32)
                    num = 0
                    for c in range(1, num_component):
                        p = (component == c)
                        if p.sum() > minarea:
                            predictions[p] = 1
                            num += 1
                    batch_preds.append(predictions)
                dice = compute_dice_class(torch.from_numpy(np.array(batch_preds)), masks_oneclasses)
                dices_range[index_threshold, index_minarea] = dice
        return dices_range


def get_model(model_name, load_path):

    model = Model(model_name).create_model()
    Solver(model).load_checkpoint(load_path)
    return model


if __name__ == "__main__":
    config = get_seg_config()
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    mask_only = True

    dataloaders = provider(config.dataset_root, os.path.join(config.dataset_root, 'train.csv'), mean, std, config.batch_size, config.num_workers, config.n_splits, mask_only)
    results = {}

    model_path = os.path.join(config.save_path, config.model_name)
    best_thresholds_sum, best_minareas_sum, max_dices_sum = [0 for x in range(len(dataloaders))], \
                                                            [0 for x in range(len(dataloaders))], [0 for x in range(len(dataloaders))]
    for fold_index, [train_loader, valid_loader] in enumerate(dataloaders):
        if fold_index != 1:
            continue

        load_path = os.path.join(model_path, '%s_fold%d_best.pth' % (config.model_name, fold_index))
        # Path to store weight + file name
        model = get_model(config.model_name, load_path)
        mychoose_threshold_minarea = ChooseThresholdMinArea(model, config.model_name, valid_loader, fold_index, model_path)
        best_thresholds, best_minareas, max_dices = mychoose_threshold_minarea.choose_threshold_minarea()
        result = {'best_thresholds': best_thresholds, 'best_minareas': best_minareas, 'max_dices': max_dices}
        results[str(fold_index)] = result

        best_thresholds_sum = [x+y for x, y in zip(best_thresholds_sum, best_thresholds)]
        best_minareas_sum = [x+y for x, y in zip(best_minareas_sum, best_minareas)]
        max_dices_sum = [x+y for x, y in zip(max_dices_sum, max_dices)]
    best_thresholds_average, best_minareas_average, max_dices_average = [x/len(dataloaders) for x in best_thresholds_sum], \
                                                                        [x/len(dataloaders) for x in best_minareas_sum], [x/len(dataloaders) for x in max_dices_sum]
    results['mean'] = {'best_thresholds': best_thresholds_average, 'best_minareas': best_minareas_average, 'max_dices': max_dices_average}
    with codecs.open(model_path + '/%s_result.json' % (config.model_name), 'w', "utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False)

    print('save the result')
