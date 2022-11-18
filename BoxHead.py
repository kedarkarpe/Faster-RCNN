import torch
import torch.nn.functional as F
import torchvision.ops
import numpy as np
from torch import nn
from torchvision import ops
from utils import *


class BoxHead(torch.nn.Module):
    def __init__(self, device='cuda', Classes=3, P=7):
        super(BoxHead,self).__init__()
        self.device = device
        self.C = Classes
        self.P = P
        self.celoss=nn.CrossEntropyLoss()
        self.smoothl1 = nn.SmoothL1Loss(reduction ='sum')
        # TODO initialize BoxHead
        # self.intermediate = nn.Sequential(
        #     nn.Linear(in_features=256*self.P*self.P, out_features = 1024),
        #     nn.ReLU(),
        #     nn.Linear(in_features=1024, out_features = 1024),
        #     nn.ReLU())
        
        # self.box_head = nn.Sequential( 
        #     nn.Linear(in_features=1024, out_features = self.C+1)
        # )

        # self.regressor_head = nn.Sequential( 
        #     nn.Linear(in_features=1024, out_features = 4*self.C)
        # )
        self.intermediate = torch.nn.Sequential(
                            torch.nn.Linear(256*self.P**2, 1024),
                            torch.nn.ReLU(),
                            torch.nn.Linear(1024, 1024),
                            torch.nn.ReLU())
        
        # Not using Softmax since we will Cross-entropy
        self.classifier = torch.nn.Linear(1024, self.C+1)

        # for each proposals it predicts one bounding box for each class!
        self.regressor = torch.nn.Linear(1024, 4*self.C)
        
    #  This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
    #  Input:
    #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #  Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
    #       labels: (total_proposals,1) (the class that the proposal is assigned)
    #       regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
    def create_ground_truth_pk(self, proposals, gt_labels, bbox):
        labels = torch.zeros((0, 1))
        regressor_target = torch.zeros((0, 4))

        for sample in range(len(gt_labels)):
            prop = proposals[sample].detach().cpu()
            box = torch.from_numpy(bbox[sample])

            iou_grid = ops.box_iou(prop, box)

            good_props = torch.max(iou_grid, dim=1)[0] > 0.5
            good_props_cls = torch.max(iou_grid, dim=1)[1][good_props]
            good_props_idx = good_props.nonzero().flatten()

            sample_cls = torch.zeros((len(prop), 1), dtype=torch.int)
            sample_cls[good_props_idx, 0] = gt_labels[sample][good_props_cls].type(torch.int)

            sample_props = torch.zeros((len(prop), 4))

            xa = ((prop[:, 0] + prop[:, 2]) / 2)[good_props_idx]
            ya = ((prop[:, 1] + prop[:, 3]) / 2)[good_props_idx]
            wa = (prop[:, 2] - prop[:, 0])[good_props_idx]
            ha = (prop[:, 3] - prop[:, 1])[good_props_idx]

            xb = (box[good_props_cls, 0] + box[good_props_cls, 2])
            yb = (box[good_props_cls, 1] + box[good_props_cls, 3])
            wb = (box[good_props_cls, 2] - box[good_props_cls, 0])
            hb = (box[good_props_cls, 3] - box[good_props_cls, 1])

            sample_props[good_props_idx, 0] = ((xb - xa) / wa)
            sample_props[good_props_idx, 1] = ((yb - ya) / ha)
            sample_props[good_props_idx, 2] = torch.log(wb / wa)
            sample_props[good_props_idx, 3] = torch.log(hb / ha)

            labels = torch.vstack((labels, sample_cls))
            regressor_target = torch.vstack((regressor_target, sample_props))

        return labels, regressor_target

    def create_ground_truth(self, proposals, gt_labels, bbox):
        labels = torch.empty(0)
        regressor_target = torch.zeros((0, 4))

        for sample in range(len(gt_labels)):
            prop = proposals[sample].detach().cpu()
            box = torch.from_numpy(bbox[sample])
            label = torch.from_numpy(gt_labels[sample])
            
            iou_grid = ops.box_iou(prop, box)

            good_props = torch.max(iou_grid, dim=1)[0] > 0.5
            good_props_cls = torch.max(iou_grid, dim=1)[1][good_props]
            good_props_idx = good_props.nonzero().flatten()

            sample_cls = torch.zeros(len(prop), dtype=torch.int)
            if len(good_props_idx) > 0:
                sample_cls[good_props_idx] = (label[good_props_cls]).type(torch.int)

            sample_props = torch.zeros((len(prop), 4))

            if len(good_props_idx) > 0:
                xa = ((prop[:, 0] + prop[:, 2]) / 2)[good_props_idx]
                ya = ((prop[:, 1] + prop[:, 3]) / 2)[good_props_idx]
                wa = (prop[:, 2] - prop[:, 0])[good_props_idx]
                ha = (prop[:, 3] - prop[:, 1])[good_props_idx]

                xb = (box[good_props_cls, 0] + box[good_props_cls, 2])
                yb = (box[good_props_cls, 1] + box[good_props_cls, 3])
                wb = (box[good_props_cls, 2] - box[good_props_cls, 0])
                hb = (box[good_props_cls, 3] - box[good_props_cls, 1])

                sample_props[good_props_idx, 0] = ((xb - xa) / wa)
                sample_props[good_props_idx, 1] = ((yb - ya) / ha)
                sample_props[good_props_idx, 2] = torch.log(wb / wa)
                sample_props[good_props_idx, 3] = torch.log(hb / ha)

            labels = torch.hstack((labels, sample_cls))

            regressor_target = torch.vstack((regressor_target, sample_props))

        labels = []
        regressor_target=[]
        for batch_id in range(len(proposals)):
            bbox_batch_id=torch.from_numpy(bbox[batch_id])
            gt_labels_batch_id=torch.from_numpy(gt_labels[batch_id])
            iou= torchvision.ops.box_iou(proposals[batch_id], bbox_batch_id)                     
            box    = torch.zeros((proposals[batch_id].shape[0],4))
            max_iou, gt_box_index = torch.max(iou, dim=1)             
            lables_batch = torch.zeros(proposals[batch_id].shape[0]).float()                 
            lables_batch[max_iou>0.5] =  gt_labels_batch_id[gt_box_index[max_iou>0.5]].float()     
            gt_box = bbox_batch_id[gt_box_index,:]
            box[:,0] = ((gt_box[:,0] + gt_box[:,2])/2 - (proposals[batch_id][:,0]+proposals[batch_id][:,2])/2)/(proposals[batch_id][:,2]-proposals[batch_id][:,0])
            box[:,1] = ((gt_box[:,1] + gt_box[:,3])/2 - (proposals[batch_id][:,1]+proposals[batch_id][:,3])/2)/(proposals[batch_id][:,3]-proposals[batch_id][:,1])
            box[:,2] = torch.log((gt_box[:,2] - gt_box[:,0])/ (proposals[batch_id][:,2] - proposals[batch_id][:,0]))
            box[:,3] = torch.log((gt_box[:,3] - gt_box[:,1])/ (proposals[batch_id][:,3] - proposals[batch_id][:,1]))
            
            labels.append(lables_batch)
            regressor_target.append(box)
            
        labels = torch.hstack(labels).reshape(-1,1)                      
        regressor_target = torch.vstack(regressor_target)  
        # labels = labels.reshape(-1, 1)
        return labels, regressor_target

    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
    def MultiScaleRoiAlign(self, fpn_feat_list, proposals, P=7):
        batch_size = len(proposals)
        feature_vectors = torch.zeros((0, 256 * P * P)).to('cuda')

        for sample in range(batch_size):
            for prop in range(len(proposals[sample])):
                w = proposals[sample][prop, 2] - proposals[sample][prop, 0]
                h = proposals[sample][prop, 3] - proposals[sample][prop, 1]

                prop_fpn_lvl = int(torch.floor(4 + torch.log(torch.sqrt((w * h)) / 224)))
                prop_fpn_lvl = max(2, prop_fpn_lvl)
                prop_fpn_lvl = min(5, prop_fpn_lvl)
                prop_fpn_lvl -= 2

                scale_x = 1088 / fpn_feat_list[prop_fpn_lvl].shape[3]
                scale_y = 800 / fpn_feat_list[prop_fpn_lvl].shape[2]

                pbox = proposals[sample][prop].reshape(1, -1).clone()
                pbox[:, 0] /= scale_x
                pbox[:, 1] /= scale_y
                pbox[:, 2] /= scale_x
                pbox[:, 3] /= scale_y

                feature_input = fpn_feat_list[prop_fpn_lvl][sample].unsqueeze(0)

                vector = torchvision.ops.roi_align(feature_input, [pbox.to('cuda')], P).flatten()
                feature_vectors = torch.vstack((feature_vectors, vector))

        return feature_vectors

    # This function does the post processing for the results of the Box Head for a batch of images
    # Use the proposals to distinguish the outputs from each image
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
    #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
    #       conf_thresh: scalar
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500,
                               keep_num_postNMS=50):
        class_logits = class_logits.cpu()
        box_regression = box_regression.cpu()

        batch_size = len(proposals)
        total_prop = len(box_regression)

        # Crop cross-boundary boxes
        for j in range(self.C + 1):
            box_regression[:, (4 * j) + 0] = torch.clamp(box_regression[:, (4 * j) + 0], min=0, max=1088)
            box_regression[:, (4 * j) + 1] = torch.clamp(box_regression[:, (4 * j) + 1], min=0, max=800)
            box_regression[:, (4 * j) + 2] = torch.clamp(box_regression[:, (4 * j) + 2], min=0, max=1088)
            box_regression[:, (4 * j) + 3] = torch.clamp(box_regression[:, (4 * j) + 3], min=0, max=800)

        # Get class with max confidence
        all_conf, all_clss = torch.max(class_logits, dim=-1)

        # Make Confidence of all background classes zero
        all_conf[all_clss == 0] = 0

        cols_to_idx = torch.linspace(4 * (all_clss - 1), 4 * (all_clss - 1) + 3, 4).T
        rows = torch.arange(total_prop).reshape(-1, 1)
        bbox_reg = box_regression[rows, cols_to_idx]

        boxes = []
        scores = []
        labels = []

        start_idx = 0
        end_idx = 0
        for i in range(batch_size):
            end_idx += len(proposals[i])

            # Get proposals for each image
            bbox_sample = bbox_reg[start_idx:end_idx]
            bbox_sample = output_decoding(bbox_sample, proposals[i].to('cpu'))
            conf_sample = all_conf[start_idx:end_idx]
            clss_sample = all_clss[start_idx:end_idx]

            conf_sorted, idx_sorted = torch.sort(conf_sample, descending=True)
            good_idx = conf_sorted > conf_thresh

            # Get high confidence proposals
            conf_sorted = conf_sorted[good_idx]
            idx_sorted = idx_sorted[good_idx]

            # Get top k proposals
            top_k = min(len(conf_sorted), keep_num_preNMS)
            idx_sorted = idx_sorted[:top_k]
            conf_sorted = conf_sorted[:top_k]
            bbox_sorted = bbox_sample[idx_sorted]
            labels_sorted = clss_sample[idx_sorted]

            iou_grid = ops.box_iou(bbox_sorted.to(self.device), bbox_sorted.to(self.device)).triu(diagonal=1)

            nms_idx = (iou_grid > 0.65).sum(dim=0) == 0

            nms_scores = conf_sorted[nms_idx]
            nms_boxes = bbox_sorted[nms_idx]
            nms_labels = labels_sorted[nms_idx]

            if nms_scores.shape[0] > keep_num_postNMS:
                boxes.append(nms_boxes[:keep_num_postNMS])
                scores.append(nms_scores[:keep_num_postNMS])
                labels.append(nms_labels[:keep_num_postNMS])
            else:
                boxes.append(nms_boxes)
                scores.append(nms_scores)
                labels.append(nms_labels)

            start_idx = end_idx

        return boxes, scores, labels

    # Compute the total loss of the classifier and the regressor
    # Input:
    #      class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
    #      box_preds: (total_proposals,4*C)      (as outputed from forward)
    #      labels: (total_proposals,1)
    #      regression_targets: (total_proposals,4)
    #      l: scalar (weighting of the two losses)
    #      effective_batch: scalar
    # Output:
    #      loss: scalar
    #      loss_class: scalar
    #      loss_regr: scalar
    def loss_clas(self,gt_clas,out_clas):
      loss_c = self.celoss(out_clas, gt_clas)
      return loss_c 

    def loss_reg(self,gt_clas,gt_coord, out_clas, out_coord):
      loss_r = 0
      self.nreg = len(gt_clas)
      for i in range(3):
        idxs = (gt_clas==i+1).nonzero()[:,0]
        pos_out_r = out_coord[idxs, i*4:(i*4)+4]
        pos_target_coord = gt_coord[idxs, :]
        loss_r=  loss_r + self.smoothl1(pos_out_r, pos_target_coord)
      return loss_r / (self.nreg + 1e-5)

    def compute_loss_ot(self,class_logits, box_preds, labels, regression_targets,l=1,effective_batch=150):
      loss = nn.CrossEntropyLoss()
      reg_loss = nn.SmoothL1Loss()
      labels = labels.type(torch.LongTensor).to('cuda').squeeze()
      label_no_back = (labels!=0).nonzero().squeeze()
      label_back = (labels==0).nonzero().squeeze()
      vals = min(label_no_back.shape[0], int(effective_batch*3/4))
      vals_back = min(label_back.shape[0], int(effective_batch*1/4))
      class_inds = torch.randperm(label_no_back.shape[0])[:vals] #get permuted range of values
      class_inds = label_no_back[class_inds] #get indices where its non zero
      class_inds_back = torch.randperm(label_back.shape[0])[:vals_back]
      class_inds_back = label_back[class_inds_back]
      class_pvals, gt_pvals =  class_logits[class_inds], labels[class_inds] #use it to get the actual non-zero labels
      box_preds_noback, gtbox_preds_noback = box_preds[class_inds,:], regression_targets[class_inds,:].to('cuda')
      class_pvals_back, gt_pvals_back =  class_logits[class_inds_back], labels[class_inds_back]
      
      loss_class_noback = loss(class_pvals, gt_pvals)
      loss_class_back = loss(class_pvals_back, gt_pvals_back)
      loss_class = loss_class_noback+loss_class_back

      get_corrds_low, get_corrds_high = ((gt_pvals-1)*4).int(), ((gt_pvals-1)*4+4).int()

      loss_regr = 0
      for i in range(get_corrds_low.shape[0]):
        pred_box = box_preds_noback[:, get_corrds_low[i]:get_corrds_high[i]]
        act_box = gtbox_preds_noback[i]
        # print()
        loss_regr += reg_loss(pred_box, act_box)
      
      loss_regr /= get_corrds_low.shape[0]
      loss = l*loss_regr + loss_class
    #   nobag_idx = (labels[:,0]).nonzero()
    #   bag_idx = (labels[:,0] == 0).nonzero()

    #   if len(nobag_idx)>(3*effective_batch/4):
    #     rand_idx = torch.randperm(len(nobag_idx))[:int(3*effective_batch/4)]
    #     sampled_nobag_idx = nobag_idx[rand_idx]
    #     rand_idx = torch.randperm(len(bag_idx))[:int(effective_batch/4)]
    #     sampled_bag_idx = bag_idx[rand_idx]
    #   else:
    #     sampled_nobag_idx = nobag_idx
    #     rand_idx = torch.randperm(len(bag_idx))[:effective_batch - len(nobag_idx)]
    #     sampled_bag_idx = bag_idx[rand_idx]

    #   out_clas = torch.squeeze(torch.cat((class_logits[sampled_nobag_idx,:], class_logits[sampled_bag_idx,:]),0))
    #   gt_clas = torch.squeeze(torch.cat((labels[sampled_nobag_idx,0], labels[sampled_bag_idx,0]),0))

    #   loss_class = self.loss_clas(gt_clas.long() ,out_clas)

    #   out_coord = torch.squeeze(box_preds[sampled_nobag_idx, :])
    #   gt_coord = torch.squeeze(regression_targets[sampled_nobag_idx, :])
    #   out_clas = torch.squeeze(class_logits[sampled_nobag_idx, :])
    #   gt_clas = torch.squeeze(labels[sampled_nobag_idx, 0])

    #   loss_regr = l * self.loss_reg(gt_clas, gt_coord, out_clas, out_coord)
      return loss_class+loss_regr, loss_class, loss_regr


    def compute_loss(self, class_logits, box_preds, labels, regression_targets, l=1, effective_batch=150):
        condn = torch.where(labels > 0)
        condition = labels[condn[0], condn[1]]

        neg_condn = torch.where(labels == 0)
        neg_condition = labels[neg_condn[0], neg_condn[1]]

        positive_labels = len(condition)
        negative_labels = len(labels) - positive_labels

        if positive_labels < (3 * effective_batch) / 4:
            negative_samples = effective_batch - positive_labels
            positive_index = torch.arange(positive_labels)
            negative_index = np.random.choice(negative_labels, negative_samples, replace=False)
        else:
            positive_index = np.random.choice(positive_labels, (3 * effective_batch) // 4, replace=False)
            negative_index = np.random.choice(negative_labels, effective_batch//4, replace=False)

        class_pred_pos = class_logits[condn[0], condn[1]][positive_index]
        class_pred_neg = class_logits[neg_condn[0], neg_condn[1]][negative_index]
        pred_class = torch.cat((class_pred_pos, class_pred_neg))

        class_gt_pos = condition[positive_index]
        class_gt_neg = neg_condition[negative_index]
        gt_class = torch.cat((class_gt_pos, class_gt_neg))

        criterion_1 = torch.nn.CrossEntropyLoss()
        loss_class = criterion_1(pred_class, gt_class)/effective_batch

        box_positive = box_preds[condn[0]][positive_index]
        idx_pos_labels = (class_gt_pos - 1) * 4
        if idx_pos_labels.is_cuda:
            idx_pos_labels = idx_pos_labels.cpu().numpy()

        one_hot_cols = np.linspace(idx_pos_labels, idx_pos_labels + 3, 4).T

        one_hot_rows = np.arange(len(box_positive)).reshape((-1, 1))

        box_predictions = box_positive[one_hot_rows, one_hot_cols].to(self.device)
        box_gt = regression_targets[condn[0]][positive_index].to(self.device)

        criterion_2 = torch.nn.SmoothL1Loss(reduction='sum')
        loss_regr = criterion_2(box_predictions, box_gt)
        loss_regr = torch.nan_to_num(loss_regr)/len(box_positive)

        loss = (loss_class + loss_regr)
        # M = effective_batch
        # n_neg = (labels[:,0] == 0).sum().item()
        # n_pos = labels.shape[0] - n_neg
        #print("Neg:",n_neg)
        #print("pos:",n_pos)
        # if n_pos < (3*M/4):
        #   num_neg_sample       = M - n_pos
        #   neg_idx              = np.random.choice(n_neg, num_neg_sample, replace = False)
        #   pos_idx              = torch.arange(n_pos)
          
        # else:
        #   pos_idx              = np.random.choice(n_pos, (3*M)//4, replace = False)
        #   neg_idx              = np.random.choice(n_neg, M - ((3*M)//4), replace = False)

        # p_class_pred           =     class_logits[(labels[:,0] != 0) , :][pos_idx,:]
        # n_class_pred           =     class_logits[(labels[:,0] == 0) , :][neg_idx,:]
        # p_label                =     labels[ (labels[:,0] != 0) ,:][pos_idx,:]
        # n_label                =     labels[ (labels[:,0] == 0) ,:][neg_idx,:]

        # class_pred             =     torch.vstack((p_class_pred,n_class_pred))
        # class_gt               =     torch.vstack((p_label,n_label))

        # loss_cp              =     nn.CrossEntropyLoss()
        # #loss_cp              =     nn.NLLLoss()
        # #loss_class           =     loss_cp(torch.log(class_pred), class_gt[:,0])
        # loss_class           =     loss_cp(class_pred, class_gt[:,0])
        
        # p_box_pred           =     box_preds[(labels[:,0] != 0) , :][pos_idx,:]          #(no of +ve samples , 4*C)

        # p_labels_to_idx      =     (p_label - 1)*4
        # if p_labels_to_idx.is_cuda:
        #   p_labels_to_idx = p_labels_to_idx.cpu().numpy()

        # col_to_idx           =     np.linspace(p_labels_to_idx,p_labels_to_idx+3,4).T
        # rows                 =     np.arange(p_box_pred.shape[0]).reshape(-1,1)

        # box_pred             =     p_box_pred[rows,col_to_idx]
        # box_gt               =     regression_targets[(labels[:,0] != 0) , :][pos_idx,:]

        # #print("box_pred", box_pred)
        # #print("box_gt", box_gt)

        # L1_loss              =     torch.nn.SmoothL1Loss(reduction = 'sum')
        # loss_regr            =     L1_loss(box_pred.squeeze(0),box_gt) 

        # loss = loss_class + l*loss_regr 

        return loss, loss_class, loss_regr

    # Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
    # Input:
    #        feature_vectors: (total_proposals, 256*P*P)
    # Outputs:
    #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
    #                                               CrossEntropyLoss you should not pass the output through softmax here)
    #        box_pred:     (total_proposals,4*C)
    def forward(self, feature_vectors):

        X = self.intermediate(feature_vectors)
        class_logits = self.classifier(X)
        box_pred = self.regressor(X)

        return class_logits, box_pred


# if __name__ == '__main__':
