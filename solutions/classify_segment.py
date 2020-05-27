import json
import os
import cv2
import torch
import numpy as np
from solutions.solver import Solver
from models.model import Model, ClassifyResNet


def post_process(probability, threshold, min_size):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def get_thresholds_minareas(json_path, fold=None):
    '''
     Obtain the optimal pixel threshold and optimal minimum connected domain of
     a specific fold of each category or the average optimal pixel threshold
     and average optimal minimum connected domain of all folds
    '''
    with open(json_path, encoding='utf-8') as json_file:
        result = json.load(json_file)
    if fold != None:
        thresholds, minareas = result[str(fold)]['best_thresholds'], result[str(fold)]['best_minareas']
    else:
        thresholds, minareas = result['mean']['best_thresholds'], result['mean']['best_minareas']
    return thresholds, minareas


class Get_Classify_Results():
    def __init__(self, model_name, fold, model_path, class_num=4, tta_flag=False):
        self.model_name = model_name
        self.fold = fold
        self.model_path = model_path
        self.class_num = class_num
        self.tta_flag = tta_flag
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.classify_model = ClassifyResNet(model_name, encoder_weights=None)
        if torch.cuda.is_available():
            self.classify_model = torch.nn.DataParallel(self.classify_model)

        self.classify_model.to(self.device)

        self.classify_model_path = os.path.join(self.model_path, '%s_classify_fold%d_best.pth' % (self.model_name, self.fold))
        self.solver = Solver(self.classify_model)
        self.classify_model = self.solver.load_checkpoint(self.classify_model_path)
        self.classify_model.eval()

    def get_classify_results(self, images, thrshold=0.5):
        if self.tta_flag:
            predict_classes = self.solver.tta(images, seg=False)
        else:
            predict_classes = self.solver.forward(images)
        predict_classes = predict_classes > thrshold
        return predict_classes


class Get_Segment_Results():
    def __init__(self, model_name, fold, model_path, class_num=4, tta_flag=False):
        self.model_name = model_name
        self.fold = fold
        self.model_path = model_path
        self.class_num = class_num
        self.tta_flag = tta_flag

        self.segment_model = Model(self.model_name, encoder_weights=None).create_model()
        self.segment_model_path = os.path.join(self.model_path, '%s_fold%d_best.pth' % (self.model_name, self.fold))
        self.solver = Solver(self.segment_model)
        self.segment_model = self.solver.load_checkpoint(self.segment_model_path)
        self.segment_model.eval()

        self.json_path = os.path.join(self.model_path, '%s_result.json' % self.model_name)
        self.best_thresholds, self.best_minareas = get_thresholds_minareas(self.json_path, self.fold)

    def get_segment_results(self, images, process_flag=True):
        if self.tta_flag:
            predict_masks = self.solver.tta(images)
        else:
            predict_masks = self.solver.forward(images)
        if process_flag:
            for index, predict_masks_classes in enumerate(predict_masks):
                for each_class, pred in enumerate(predict_masks_classes):
                    pred_binary, _ = post_process(pred.detach().cpu().numpy(), self.best_thresholds[each_class], self.best_minareas[each_class])
                    predict_masks[index, each_class] = torch.from_numpy(pred_binary)
        return predict_masks


class Classify_Segment_Fold():
    def __init__(self, classify_fold, seg_fold, model_path, class_num=4, tta_flag=False, kaggle=0):

        self.classify_fold = classify_fold
        self.seg_fold = seg_fold
        self.model_path = model_path
        self.class_num = class_num
        for (model_name, fold) in self.classify_fold.items():
            if kaggle == 0:
                pth_path = os.path.join(self.model_path, model_name)
            else:
                pth_path = self.model_path                
            self.classify_model = Get_Classify_Results(model_name, fold, pth_path, self.class_num, tta_flag=tta_flag)
        for (model_name, fold) in self.classify_fold.items():
            if kaggle == 0:
                pth_path = os.path.join(self.model_path, model_name)
            else:
                pth_path = self.model_path
            self.segment_model = Get_Segment_Results(model_name, fold, pth_path, self.class_num, tta_flag=tta_flag)

    def classify_segment(self, images):

        predict_classes = self.classify_model.get_classify_results(images)
        predict_masks = self.segment_model.get_segment_results(images)
        for index, predicts in enumerate(predict_classes):
            for each_class, pred in enumerate(predicts):
                if pred == 0:
                    predict_masks[index, each_class, ...] = 0
        return predict_masks


class Classify_Segment_Folds():
    def __init__(self, classify_folds, segment_folds, model_path, class_num=4, tta_flag=False, kaggle=0):

        self.classify_folds = classify_folds
        self.segment_folds = segment_folds
        self.model_path = model_path
        self.class_num = class_num
        self.tta_flag = tta_flag
        self.kaggle = kaggle

        self.classify_models, self.segment_models = list(), list()
        self.get_classify_segment_models()

    def get_classify_segment_models(self):

        for (model_name, fold) in self.classify_folds.items():
            if self.kaggle == 0:
                pth_path = os.path.join(self.model_path, model_name)
            else:            
                pth_path = self.model_path                
            self.classify_models.append(Get_Classify_Results(model_name, fold, pth_path, self.class_num, tta_flag=self.tta_flag))
        for (model_name, fold) in self.segment_folds.items():
            if self.kaggle == 0:
                pth_path = os.path.join(self.model_path, model_name)
            else:            
                pth_path = self.model_path
            self.segment_models.append(Get_Segment_Results(model_name, fold, pth_path, self.class_num, tta_flag=self.tta_flag))

    def classify_segment_folds(self, images):

        results = torch.zeros(images.shape[0], self.class_num, images.shape[2], images.shape[3])
        for classify_model, segment_model in zip(self.classify_models, self.segment_models):

            predict_classes = classify_model.get_classify_results(images)

            predict_masks = segment_model.get_segment_results(images)
            for index, predicts in enumerate(predict_classes):
                for each_class, pred in enumerate(predicts):
                    if pred == 0:
                        predict_masks[index, each_class, ...] = 0
            results += predict_masks.detach().cpu()
        vote_model_num = len(self.segment_folds)
        vote_ticket = round(vote_model_num / 2.0)
        results = results > vote_ticket

        return results


class Classify_Segment_Folds_Split():
    def __init__(self, classify_folds, segment_folds, model_path, class_num=4, tta_flag=False, kaggle=0):

        self.classify_folds = classify_folds
        self.segment_folds = segment_folds
        self.model_path = model_path
        self.class_num = class_num
        self.tta_flag = tta_flag
        self.kaggle = kaggle

        self.classify_models, self.segment_models = list(), list()
        self.get_classify_segment_models()

    def get_classify_segment_models(self):

        for (model_name, fold) in self.classify_folds.items():
            if self.kaggle == 0:
                pth_path = os.path.join(self.model_path, model_name)
            else:            
                pth_path = self.model_path                
            self.classify_models.append(Get_Classify_Results(model_name, fold, pth_path, self.class_num, tta_flag=self.tta_flag))
        for (model_name, fold) in self.segment_folds.items():
            if self.kaggle == 0:
                pth_path = os.path.join(self.model_path, model_name)
            else:            
                pth_path = self.model_path
            self.segment_models.append(Get_Segment_Results(model_name, fold, pth_path, self.class_num, tta_flag=self.tta_flag))

    def classify_segment_folds(self, images, average_strategy=False):

        classify_results = torch.zeros(images.shape[0], self.class_num)
        segment_results = torch.zeros(images.shape[0], self.class_num, images.shape[2], images.shape[3])

        for classify_index, classify_model in enumerate(self.classify_models):
            classify_result_fold = classify_model.get_classify_results(images)
            classify_results += classify_result_fold.detach().cpu().squeeze().float()
        classify_vote_model_num = len(self.classify_folds)
        classify_vote_ticket = round(classify_vote_model_num / 2.0)
        classify_results = classify_results > classify_vote_ticket

        if average_strategy:
            for segment_index, segment_model in enumerate(self.segment_models):
                segment_result_fold = segment_model.get_segment_results(images, process_flag=False)
                segment_results += segment_result_fold.detach().cpu()
            average_thresholds, average_minareas = get_thresholds_minareas(os.path.join(self.model_path, 'result.json'))
            segment_results = segment_results/len(self.segment_folds)
            for index, predict_masks_classes in enumerate(segment_results):
                for each_class, pred in enumerate(predict_masks_classes):
                    pred_binary, _ = post_process(pred.detach().cpu().numpy(),
                                                  average_thresholds[each_class],
                                                  average_minareas[each_class])
                    segment_results[index, each_class] = torch.from_numpy(pred_binary)

        else:
            for segment_index, segment_model in enumerate(self.segment_models):
                segment_result_fold = segment_model.get_segment_results(images)
                segment_results += segment_result_fold.detach().cpu()
            segment_vote_model_num = len(self.segment_folds)
            segment_vote_ticket = round(segment_vote_model_num / 2.0)
            segment_results = segment_results > segment_vote_ticket

        for batch_index, classify_result in enumerate(classify_results):
            segment_results[batch_index, ~classify_result, ...] = 0

        return segment_results


class Segment_Folds():
    def __init__(self, model_name, n_splits,
                 model_path, class_num=4, tta_flag=False):

        self.model_name = model_name
        self.n_splits = n_splits
        self.model_path = model_path
        self.class_num = class_num
        self.tta_flag = tta_flag

        self.segment_models = list()
        self.get_segment_models()

    def get_segment_models(self):
        for fold in self.n_splits:
            self.segment_models.append(Get_Segment_Results(self.model_name,
                                                           fold,
                                                           self.model_path,
                                                           self.class_num,
                                                           tta_flag=self.tta_flag))

    def segment_folds(self, images):

        results = torch.zeros(images.shape[0],
                              self.class_num,
                              images.shape[2],
                              images.shape[3])
        for segment_model in self.segment_models:

            predict_masks = segment_model.get_segment_results(images)
            results += predict_masks.detach().cpu()
        vote_model_num = len(self.n_splits)
        vote_ticket = round(vote_model_num / 2.0)
        results = results > vote_ticket

        return results