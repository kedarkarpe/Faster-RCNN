import torch
from torch.nn import functional as F
from torchvision import ops
from torch import nn, Tensor
from dataset import *
from utils import *



class RPNHead(torch.nn.Module):

    def __init__(self, device='cpu', anchors_param=dict(ratio=1, scale=400, grid_size=(50, 68), stride=16)):

        # Initialize the backbone, intermediate layer clasifier and regressor heads of the RPN
        super(RPNHead, self).__init__()

        self.device = device
        # Define Backbone
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=1, padding='same',
                                     bias=False)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=1, padding='same',
                                     bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding='same',
                                     bias=False)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=1, padding='same',
                                     bias=False)
        self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), stride=1, padding='same',
                                     bias=False)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.batch1 = torch.nn.BatchNorm2d(num_features=16)
        self.batch2 = torch.nn.BatchNorm2d(num_features=32)
        self.batch3 = torch.nn.BatchNorm2d(num_features=64)
        self.batch4 = torch.nn.BatchNorm2d(num_features=128)
        self.batch5 = torch.nn.BatchNorm2d(num_features=256)
        self.relu = torch.nn.ReLU()

        # Define Intermediate Layer
        self.conv6 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding='same',
                                     bias=False)
        self.batch6 = torch.nn.BatchNorm2d(num_features=256)

        # Define Proposal Classifier Head
        self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1, 1), stride=1, padding='same',
                                     bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        # Define Proposal Regressor Head
        self.regressor_head = torch.nn.Conv2d(in_channels=256, out_channels=4, kernel_size=(1, 1), stride=1,
                                              padding='same', bias=False)

        # find anchors
        self.anchors_param = anchors_param
        self.anchors = self.create_anchors(self.anchors_param['ratio'], self.anchors_param['scale'],
                                           self.anchors_param['grid_size'], self.anchors_param['stride'])
        self.ground_dict = {}

    # Forward  the input through the backbone the intermediate layer and the RPN heads
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       logits: (bz,1,grid_size[0],grid_size[1])}
    #       bbox_regs: (bz,4, grid_size[0],grid_size[1])}
    def forward(self, X):
        # forward through the Backbone
        X = self.maxpool(self.relu(self.batch1(self.conv1(X))))
        X = self.maxpool(self.relu(self.batch2(self.conv2(X))))
        X = self.maxpool(self.relu(self.batch3(self.conv3(X))))
        X = self.maxpool(self.relu(self.batch4(self.conv4(X))))
        X = self.relu(self.batch5(self.conv5(X)))

        # forward through the Intermediate layer
        X = self.relu(self.batch6(self.conv6(X)))

        # forward through the Classifier Head
        logits = self.sigmoid(self.conv7(X))

        # forward through the Regressor Head
        bbox_regs = self.regressor_head(X)

        assert logits.shape[1:4] == (1, self.anchors_param['grid_size'][0], self.anchors_param['grid_size'][1])
        assert bbox_regs.shape[1:4] == (4, self.anchors_param['grid_size'][0], self.anchors_param['grid_size'][1])

        return logits, bbox_regs

    # Forward input batch through the backbone
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       X: (bz,256,grid_size[0],grid_size[1])
    def forward_backbone(self, X):
        # forward through the backbone
        X = self.maxpool(self.relu(self.batch1(self.conv1(X))))
        X = self.maxpool(self.relu(self.batch2(self.conv2(X))))
        X = self.maxpool(self.relu(self.batch3(self.conv3(X))))
        X = self.maxpool(self.relu(self.batch4(self.conv4(X))))
        X = self.relu(self.batch5(self.conv5(X)))
        assert X.shape[1:4] == (256, self.anchors_param['grid_size'][0], self.anchors_param['grid_size'][1])

        return X

    # This function creates the anchor boxes
    # Output:
    #       anchors: (grid_size[0], grid_size[1], 4)
    def create_anchors(self, aspect_ratio, scale, grid_sizes, stride):
        height = scale / np.sqrt(aspect_ratio)
        width = aspect_ratio * height

        y, x = torch.meshgrid(torch.arange(grid_sizes[0]), torch.arange(grid_sizes[1]))
        x = (x + 0.5) * stride
        y = (y + 0.5) * stride

        anchors = torch.zeros((grid_sizes[0], grid_sizes[1], 4))
        anchors[:, :, 0] = x
        anchors[:, :, 1] = y
        anchors[:, :, 2] = width
        anchors[:, :, 3] = height
        assert anchors.shape == (grid_sizes[0], grid_sizes[1], 4)

        return anchors

    def get_anchors(self):
        return self.anchors

    # This function creates the ground truth for a batch of images by using
    # create_ground_truth internally
    # Input:
    #      bboxes_list: list:len(bz){(n_obj,4)}
    #      indexes:      list:len(bz)
    #      image_shape:  tuple:len(2)
    # Output:
    #      ground_clas: (bz,1,grid_size[0],grid_size[1])
    #      ground_coord: (bz,4,grid_size[0],grid_size[1])
    def create_batch_truth(self, bboxes_list, indexes, image_shape):
        batch_size = len(indexes)

        ground_clas = torch.zeros(
            (batch_size, 1, self.anchors_param['grid_size'][0], self.anchors_param['grid_size'][1]))
        ground_coord = torch.zeros(
            (batch_size, 4, self.anchors_param['grid_size'][0], self.anchors_param['grid_size'][1]))

        for sample in range(batch_size):
            cls, coord, anch, bbx = self.create_ground_truth(bboxes_list[sample], indexes[sample],
                                                             self.anchors_param['grid_size'],
                                                             self.anchors, image_shape)

            ground_clas[sample] = cls
            ground_coord[sample] = coord

        assert ground_clas.shape[1:4] == (1, self.anchors_param['grid_size'][0], self.anchors_param['grid_size'][1])
        assert ground_coord.shape[1:4] == (4, self.anchors_param['grid_size'][0], self.anchors_param['grid_size'][1])

        return ground_clas, ground_coord, anch, bbx

    # This function creates the ground truth for one image
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset used for caching)
    #       grid_size:   tuple:len(2)
    #       anchors:     (grid_size[0], grid_size[1], 4)
    # Output:
    #       ground_clas:  (1, grid_size[0], grid_size[1])
    #       ground_coord: (4, grid_size[0], grid_size[1])
    def create_ground_truth(self, bboxes, index, grid_size, anchors, image_size):
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        ground_clas = -torch.ones((1, grid_size[0], grid_size[1]))
        ground_coord = torch.zeros((4, grid_size[0], grid_size[1]))

        # Find cross boundary anchors
        xmin = anchors[:, :, 0] - torch.divide(anchors[:, :, 2], 2)
        ymin = anchors[:, :, 1] - torch.divide(anchors[:, :, 3], 2)
        xmax = anchors[:, :, 0] + torch.divide(anchors[:, :, 2], 2)
        ymax = anchors[:, :, 1] + torch.divide(anchors[:, :, 3], 2)
        cross_bound = torch.logical_and(torch.logical_and(xmin >= 0, ymin >= 0), torch.logical_and(xmax < image_size[1], ymax < image_size[0]))

        anchors_xyxy = ops.box_convert(anchors.reshape((anchors.shape[0] * anchors.shape[1], anchors.shape[2])), 'cxcywh', 'xyxy')
        ops.box_iou(anchors_xyxy, torch.from_numpy(bboxes))

        # Find Positive Anchors
        pos_iou = torch.max(grid_iou, dim=-1)[0] > 0.7
        other_pos_iou = torch.logical_and(torch.max(grid_iou, dim=-1)[0] < 0.7, torch.max(grid_iou, dim=1)[0] > 0.3)

        pos_iou = torch.logical_or(pos_iou, other_pos_iou).reshape((grid_size[0], grid_size[1]))
        pos_idx = torch.logical_and(pos_iou, cross_bound).nonzero()
        ground_clas[0, pos_idx[:, 0], pos_idx[:, 1]] = 1

        max_iou = torch.max(grid_iou, dim=-1)[1].reshape((grid_size[0], grid_size[1]))
        which_bbox = max_iou[pos_idx[:, 0], pos_idx[:, 1]]

        bbx = ops.box_convert(torch.from_numpy(bboxes), 'xyxy', 'cxcywh')[which_bbox]
        anc = anchors[pos_idx[:, 0], pos_idx[:, 1]]
        ground_coord[0, pos_idx[:, 0], pos_idx[:, 1]] = torch.divide(bbx[:, 0] - anc[:, 0], anc[:, 2])
        ground_coord[1, pos_idx[:, 0], pos_idx[:, 1]] = torch.divide(bbx[:, 1] - anc[:, 1], anc[:, 3])
        ground_coord[2, pos_idx[:, 0], pos_idx[:, 1]] = torch.log(torch.divide(bbx[:, 2], anc[:, 2]))
        ground_coord[3, pos_idx[:, 0], pos_idx[:, 1]] = torch.log(torch.divide(bbx[:, 3], anc[:, 3]))

        # Find Negative Anchors
        neg_iou = torch.all(grid_iou.reshape((grid_size[0], grid_size[1], -1)) < 0.3, -1)
        neg_idx = torch.logical_and(neg_iou, cross_bound).nonzero()
        ground_clas[0, neg_idx[:, 0], neg_idx[:, 1]] = 0

        self.ground_dict[key] = (ground_clas, ground_coord)

        assert ground_clas.shape == (1, grid_size[0], grid_size[1])
        assert ground_coord.shape == (4, grid_size[0], grid_size[1])

        return ground_clas, ground_coord

    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self, p_out, n_out):
        # compute classifier's loss
        ones_ = torch.ones(p_out.shape[0], device=self.device)
        zeros_ = torch.zeros(n_out.shape[0], device=self.device)

        ground_truths = torch.cat((ones_, zeros_), dim=0).to(self.device)
        predictions = torch.cat((p_out, n_out), dim=0)

        criterion = torch.nn.BCELoss(reduction='sum')
        loss = criterion(predictions, ground_truths)
        sum_count = p_out.shape[0] + n_out.shape[0]

        return loss, sum_count

    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self, pos_target_coord, pos_out_r):
        # compute regressor's loss
        criterion = torch.nn.SmoothL1Loss(reduction='sum')
        loss = criterion(pos_out_r, pos_target_coord)
        sum_count = pos_target_coord.shape[0]

        return loss, sum_count

    # Compute the total loss
    # Input:
    #       clas_out: (bz,1,grid_size[0],grid_size[1])
    #       regr_out: (bz,4,grid_size[0],grid_size[1])
    #       targ_clas:(bz,1,grid_size[0],grid_size[1])
    #       targ_regr:(bz,4,grid_size[0],grid_size[1])
    #       l: lambda constant to weight between the two losses
    #       effective_batch: the number of anchors in the effective batch
    def compute_loss(self, clas_out, regr_out, targ_clas, targ_regr, l=1, effective_batch=50):
        # compute the total loss
        regr_out_flat,  class_out_flat, _ = output_flattening(regr_out, clas_out, self.anchors)
        targ_regr_flat, targ_clas_flat, _ = output_flattening(targ_regr, targ_clas, self.anchors)

        print(regr_out_flat.size(), targ_regr_flat.size())
        positive_conditions = targ_clas_flat[targ_clas_flat == 1].shape[0]
        negative_conditions = targ_clas_flat[targ_clas_flat == 0].shape[0]

        if positive_conditions < (effective_batch / 2):
            index_neg = np.random.choice(negative_conditions, effective_batch - positive_conditions, replace=False)

            p_out_class = class_out_flat[targ_clas_flat == 1]
            n_out_class = class_out_flat[targ_clas_flat == 0][index_neg]

            p_out_regr = regr_out_flat[(targ_clas_flat == 1).squeeze(), ]
            p_targ_regr = targ_regr_flat[(targ_clas_flat == 1).squeeze(), ]

        else:
            index_pos = np.random.choice(positive_conditions, int(effective_batch / 2), replace=False)
            index_neg = np.random.choice(negative_conditions, int(effective_batch / 2), replace=False)

            p_out_class = class_out_flat[targ_clas_flat == 1][index_pos]
            n_out_class = class_out_flat[targ_clas_flat == 0][index_neg]

            p_out_regr = regr_out_flat[(targ_clas_flat ==1).squeeze(), ][index_pos]
            p_targ_regr = targ_regr_flat[(targ_clas_flat ==1).squeeze(), ][index_pos]

        loss_c, sum_count_class = self.loss_class(p_out_class, n_out_class)
        loss_r, sum_count_class = self.loss_reg(p_out_regr, p_targ_regr)
        print(p_out_regr.shape, p_targ_regr.shape)
        loss = loss_c + (l * loss_r)

        return loss, loss_c, loss_r

    # Post process for the outputs for a batch of images
    # Input:
    #       out_c:  (bz,1,grid_size[0],grid_size[1])}
    #       out_r:  (bz,4,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinates of the boxes that the NMS kept)
    def postprocess(self, out_c, out_r, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10):
        # postprocess a batch of images
        nms_clas_list, nms_prebox_list, top_proposal_clas_list, top_proposal_regr = [], [], [], []

        for i in range(out_c.shape[0]):
            batch_score = self.postprocessImg(out_c[i], out_r[i], IOU_thresh, keep_num_preNMS, keep_num_postNMS)

            nms_clas_list.append(batch_score[0])
            nms_prebox_list.append(batch_score[1])

            top_proposal_clas_list.append(batch_score[2])
            top_proposal_regr.append(batch_score[3])

        return nms_clas_list, nms_prebox_list

    # Post process the output for one image
    # Input:
    #      mat_clas: (1,grid_size[0],grid_size[1])}  (scores of the output boxes)
    #      mat_coord: (4,grid_size[0],grid_size[1])} (encoded coordinates of the output boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)
    def postprocessImg(self, mat_clas, mat_coord, IOU_thresh, keep_num_preNMS, keep_num_postNMS):
        # postprocess a single image
        mat_coord_flat, mat_class_flat, mat_anchor_flat = output_flattening(mat_clas[None, :, :, :],
                                                                            mat_coord[None, :, :, :], self.get_anchors)
        box_decoded = output_decoding(mat_coord_flat, mat_anchor_flat)

        box_decoded[:, 0] = torch.where(box_decoded[:, 0] < 0, 0, box_decoded[:, 0])
        box_decoded[:, 1] = torch.where(box_decoded[:, 1] < 0, 0, box_decoded[:, 1])
        box_decoded[:, 2] = torch.where(box_decoded[:, 2] > self.image_size[0], self.image_size[0], box_decoded[:, 2])
        box_decoded[:, 3] = torch.where(box_decoded[:, 3] > self.image_size[1], self.image_size[1], box_decoded[:, 3])

        sorted_class, index = torch.sort(mat_class_flat, dim=0, descending=True)

        top_proposals_indexes = index[: keep_num_preNMS]
        top_proposed_class = mat_class_flat[top_proposals_indexes]
        top_proposed_regr = mat_coord_flat[top_proposals_indexes]

        nms_clas, nms_prebox = self.NMS(top_proposed_class.clone(), top_proposed_regr, IOU_thresh)

        return nms_clas, nms_prebox

    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self, clas, prebox, thresh):
        # perform NMS
        class_ = torch.clone(clas).squeeze().detach().cpu().numpy()

        while True:
            if np.count_nonzero(class_) == 0:
                break
            bbox = torch.clone(prebox[np.argwhere(class_ == class_.max())].squeeze())
            class_[np.argwhere(class_ == class_.max())] = 0
            iou_calc = IOU(bbox, prebox.squeeze())

            # supressing box > 0.5 iou
            class_[iou_calc > thresh and torch.abs(iou_calc - 1.0) > 0.0001] = 0
            clas[iou_calc > thresh and torch.abs(iou_calc - 1.0) > 0.0001] = 0
        nms_clas, nms_prebox = clas[clas > 0], prebox[clas > 0]

        return nms_clas, nms_prebox
