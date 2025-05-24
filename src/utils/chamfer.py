import torch
from torch.nn import functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.loss.chamfer import _validate_chamfer_reduction_inputs, _handle_pointcloud_input


def chamfer_distance(x, y, x_lengths=None, y_lengths=None, x_normals=None, y_normals=None, weights=None,
                     batch_reduction="mean", point_reduction="mean", return_L1=False, return_mean=False):
    """
    Copy from https://github.com/facebookresearch/pytorch3d repo (see pytorch3d/loss/chamfer.py)
    with following modifications to be comparable to OccNet and DVR results [Niemeyer et al., 2019]
    (https://github.com/autonomousvision/differentiable_volumetric_rendering, see im2mesh/eval.py file for details):
        - support for returning chamfer-L1 instead of chamfer-L2
        - support for mean (cham_x, cham_y) instead of sum

    Chamfer distance between two pointclouds x and y.
    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    if return_L1:
        cham_x, cham_y = cham_x.sqrt(), cham_y.sqrt()

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None
    if return_mean:
        cham_dist, cham_normals = 0.5 * cham_dist, 0.5 * cham_normals if return_normals else None
    return cham_dist, cham_normals


def generate_onehot_labels(labels):
    batch_size, num_points, dimension = labels.shape
    _,max_index = torch.max(labels,dim=2)
    one_hot_output = torch.zeros_like(labels).scatter_(2,max_index.unsqueeze(-1),1)
    return one_hot_output

def separate_pointcloud(points,labels):
    batch_size, num_points,  _ = points.shape
    separate_points = {}
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        label_mask = labels == label
        label_mask = label_mask.view(batch_size,num_points,1).expand(batch_size,num_points,3)
        points_with_label = points[label_mask].view(-1,3)
        separate_points[label.item()] = points_with_label
    return separate_points
        

def part_chamfer_distance(x,y,x_labels,y_labels,x_lengths=None,y_lengths=None,weights=None,batch_reduction="mean",point_reduction="mean",return_L1=False,return_mean=False):
    """
    The advanced version of chamfer distance by adding the comparision of part labels
    If the predicted labels and ground truth are same, calculate their corresponding chamfer distance. 
    Chamfer distance between two pointclouds x and y.
    x: FloatTensor of shape (N,P1,D) with at most P1 points in each batch element, batch size N and feature dimension D.
    y: FloatTensor of shape (N,P2,D) with at most P2 points in each batch element, batch size N and feature dimension D. 
    x_labels: (N,P1,D1) with labels for each point
    y_labels: (N,P2,D1) with labels for each point
    
    Return:
      Loss: Mean loss for chamfer distance of different labels.
    """
    batch_size, num_points, dimension = x_labels.shape
    y_labels = generate_onehot_labels(y_labels)
    chamfer_distance_total = 0
    valid_number =0
    chamfer_distances_per_part = {}
    for index in range(batch_size):
        #unique_labels = torch.unique(x_labels.view(-1,dimension),dim=0)
        unique_labels = torch.unique(x_labels[index].view(-1,dimension),dim=0)
        chamfer_dist = 0
        N = len(unique_labels)
        unique_labels_y = torch.unique(y_labels[index].view(-1,dimension),dim=0)
        for i, label in enumerate(unique_labels):
            x_equal = torch.all(x_labels[index] == label, dim=1, keepdim=True).expand(-1,3)
            y_equal = torch.all(y_labels[index] == label, dim=1, keepdim=True).expand(-1,3)
            x_points = x[index][x_equal].view(-1,3).unsqueeze(0)
            y_points = y[index][y_equal].view(-1,3).unsqueeze(0)
            if x_points.shape[1]>0:
                valid_number += 1
            if x_points.shape[1]>0 and y_points.shape[1]>0:
                cham_dist, _ = chamfer_distance(x_points, y_points, return_L1=True, return_mean=True)
                chamfer_dist += cham_dist
                valid_number += 1
                
        #chamfer_dist /= N 
               
        chamfer_distance_total += chamfer_dist
    #chamfer_distance_total /= batch_size
    chamfer_distance_total /= valid_number
    print(valid_number)
    return chamfer_distance_total