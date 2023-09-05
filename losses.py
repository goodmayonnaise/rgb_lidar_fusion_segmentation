import torch 
from torch.autograd import Variable

class FocalLoss(nn.Module):
    """Focal loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryFocalLoss
    Return:
        same as BinaryFocalLoss
    """
    def __init__(self,  alpha:float = 0.25, gamma:float = 2, eps = 1e-8, ignore_index=0, reduction:str ='mean'):
        super(FocalLoss, self).__init__()
        self.nclasses = 20 
        self.ignore_index = ignore_index        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
       
        if probas.dim() == 4:
            # 2D segmentation
            B, C, H, W = probas.size()
            probas = probas.contiguous().permute(0, 2, 3, 1).contiguous().view(-1, C) # (B=1)*H*W, C

        if labels.dim() == 4:# B,C,H,W -> B,H,W
            labels_arg = torch.argmax(labels, dim=1)
            labels_arg = labels_arg.view(-1)# B,H,W -> B*H*W

        if labels.dim() == 4:# B,C,H,W -> B*H*W, C
            # assumes output of a sigmoid layer
            B, C, H, W = labels.size()
            labels = labels.view(B, C, H, W).permute(0, 2, 3, 1).contiguous().view(-1, C)# (B=1)*H*W, C
        
        if ignore is None:
            return probas, labels

        valid = (labels_arg != ignore)
        vprobas = probas[valid.nonzero(as_tuple=False).squeeze()] 
        vlabels = labels[valid.nonzero(as_tuple=False).squeeze()] 
        return vprobas, vlabels
    
    
    def per_class(self, t):

        per_class = torch.zeros([t.shape[0], self.nclasses, t.shape[1], t.shape[2]]).cuda()

        for i in range(self.nclasses):
            per_class[:,i] = torch.where(t==i, 1, 0)
        
        return per_class
    
    def forward(self, predicts: Tensor, targets: Tensor):     # target b h w 
        loss_total=[]
        targets = self.per_class(targets) # b h w -> b c h w
        for predict, target in zip(predicts, targets):
            predict = predict.unsqueeze(0)
            target = target.unsqueeze(0)
            
            predict, target = self.flatten_probas(predict, target, ignore=0) # (1, C, H, W) -> (K,C)

            term_true =  - self.alpha * ((1 - predict) ** self.gamma) * torch.log(predict+self.eps) 
            term_false = - (1-self.alpha) * (predict**self.gamma) * torch.log(1-predict+self.eps) 

            loss = torch.sum(term_true * target + term_false * (1-target), dim=-1)# (1*K) 
            
            loss_total.append(loss)
 
        if self.reduction == "mean":
            return torch.mean(torch.cat(loss_total))
        elif self.reduction == "sum":
            return torch.sum(torch.cat(loss_total))
        elif self.reduction == "none":
            return torch.cat(loss_total)


class Lovasz_loss(nn.Module):
    def __init__(self, nclasses=20, reduction:str='mean', ignore_index:int = 0):
        super(Lovasz_loss, self).__init__()
        self.reduction = reduction
        self.ignore_idx = ignore_index 
        self.nclasses = nclasses
    
    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """    
        def isnan(x):
            return x != x

        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1] 
        return jaccard   


    def lovasz_softmax_flat(self, probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
        probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        C = probas.size(1)#클래스 수
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if (classes == 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()# target - pred
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted)))) 
        return self.mean(losses)

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:# B,C,N -> B*N, C
            # assumes output of a sigmoid layer
            B, C, N = probas.size()
            probas = probas.view(B, C, 1, N).permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        elif probas.dim() == 5:
            # 3D segmentation
            B, C, L, H, W = probas.size()
            probas = probas.contiguous().permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        # if labels.dim() == 3:# B,C,N -> B,N
        #     labels = torch.argmax(labels, dim=1)
        # labels = labels.view(-1)# B,N -> B*N
        if ignore is None:
            return probas, labels
        labels = labels.reshape(-1)
        valid = (labels != ignore) 
        vprobas = probas[valid.nonzero(as_tuple=False).squeeze()] 
        vlabels = labels[valid]

        return vprobas, vlabels
        
    def lovasz_softmax(self, probas, labels, classes='present', per_image=True, ignore=0):
        """
        Multi-class Lovasz-Softmax loss
        probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
        """

        if per_image: # mean reduction
            loss = self.mean(self.lovasz_softmax_flat(*self.flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                             for prob, lab in zip(probas, labels))
        else: # sum reduction
            loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels, ignore), classes=classes)
        return loss


    def forward(self, uv_out, uv_label):
        lovasz_loss = self.lovasz_softmax(uv_out, uv_label, ignore=self.ignore_idx)
        return lovasz_loss

class FocalLosswithLovaszRegularizer(nn.Module):
    def __init__(self,  alpha:float = 0.75, gamma:float = 2, eps = 1e-8, smooth=1, p=2, reduction:str = 'mean', ignore_idx:int = 0):
        super(FocalLosswithLovaszRegularizer, self).__init__()
        self.reduction = reduction
        self.ignore_idx = ignore_idx
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Focal_loss = FocalLoss(alpha = self.alpha, gamma = self.gamma, eps = self.eps, ignore_index = self.ignore_idx, reduction=reduction)
        # self.Focal_loss = Focal_3D_loss(alpha = self.alpha, gamma = self.gamma, eps = self.eps, ignore_index = self.ignore_idx, reduction=reduction)
        self.Lovasze_loss = Lovasz_loss(reduction=self.reduction, ignore_index=self.ignore_idx)

    def forward(self,pred:Tensor, label:Tensor):
        # assert pred.shape == label.shape, 'predict & target shape do not match'
        # pred = F.softmax(pred, dim=1)
        f_loss = self.Focal_loss(pred, label)
        # lovasz_regularization = self.Lovasze_loss(pred*label, label)
        # return f_loss + (8 * lovasz_regularization)
        lovasz_loss = self.Lovasze_loss(pred, label)
        return f_loss + lovasz_loss
