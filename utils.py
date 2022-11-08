import torch
from functools import partial


def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)

    return tuple(map(list, zip(*map_results)))


# This function computes the IOU between two set of boxes
def IOU(boxA, boxB):
    # Get coordinates from centers, width and height
    xa1, ya1 = boxA[0] - boxA[2] / 2, boxA[1] - boxA[3] / 2
    xa2, ya2 = boxA[0] + boxA[2] / 2, boxA[1] + boxA[3] / 2

    xb1, yb1 = boxB[0] - boxB[2] / 2, boxB[1] - boxB[3] / 2
    xb2, yb2 = boxB[0] + boxB[2] / 2, boxB[1] + boxB[3] / 2

    # Get bbox
    x_top_left, y_top_left = torch.max(xa1, xb1), torch.max(ya1, yb1)
    x_bot_right, y_bot_right = torch.min(xa2, xb2), torch.min(ya2, yb2)

    # Compute Intersection
    intersection = torch.max((x_bot_right - x_top_left), torch.zeros_like(x_bot_right)) * torch.max(
        (y_bot_right - y_top_left), torch.zeros_like(y_bot_right))
    union = ((xa2 - xa1) * (ya2 - ya1)) + ((xb2 - xb1) * (yb2 - yb1)) - intersection

    iou = (intersection / union)

    return iou

# This function flattens the output of the network and the corresponding anchors
# in the sense that it concatenates  the outputs and the anchors from all the grid cells
# from all the images into 2D matrices
# Each row of the 2D matrices corresponds to a specific anchor/grid cell
# Input:
#       out_r: (bz,4,grid_size[0],grid_size[1])
#       out_c: (bz,1,grid_size[0],grid_size[1])
#       anchors: (grid_size[0],grid_size[1],4)
# Output:
#       flatten_regr: (bz*grid_size[0]*grid_size[1],4)
#       flatten_clas: (bz*grid_size[0]*grid_size[1])
#       flatten_anchors: (bz*grid_size[0]*grid_size[1],4)
def output_flattening(out_r, out_c, anchors):
    flatten_regr = torch.reshape(torch.moveaxis(out_r, 1, -1), (-1, 4))
    flatten_clas = torch.permute(out_c, (0, 2, 3, 1)).reshape((-1, 1))
    anchors = torch.repeat_interleave(torch.unsqueeze(anchors, dim=0), len(out_r), dim=0)
    flatten_anchors = anchors.reshape((-1, anchors.shape[3]))

    return flatten_regr, flatten_clas, flatten_anchors


# This function decodes the output that is given in the encoded format
# into box coordinates where it returns the upper left and lower right corner of the proposed box
# Input:
#       flatten_out: (total_number_of_anchors*bz,4)
#       flatten_anchors: (total_number_of_anchors*bz,4)
# Output:
#       box: (total_number_of_anchors*bz,4)
def output_decoding(flatten_out, flatten_anchors, device='cpu'):
    box = torch.zeros_like(flatten_out)

    x = torch.add(torch.multiply(flatten_out[:, 0], flatten_anchors[:, 2]), flatten_anchors[:, 0])
    y = torch.add(torch.multiply(flatten_out[:, 1], flatten_anchors[:, 3]), flatten_anchors[:, 1])
    w = torch.multiply(torch.exp(flatten_out[:, 2]), flatten_anchors[:, 2])
    h = torch.multiply(torch.exp(flatten_out[:, 3]), flatten_anchors[:, 3])

    box[:, 0] = x - w / 2
    box[:, 1] = y - h / 2
    box[:, 2] = x + w / 2
    box[:, 3] = y + h / 2

    return box
