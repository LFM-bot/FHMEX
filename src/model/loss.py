import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Pair-wise Noise Contrastive Estimation Loss
    """

    def __init__(self, temperature, similarity_type):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature  # temperature
        self.sim_type = similarity_type  # cos or dot
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, aug_hidden_view1, aug_hidden_view2, mask=None):
        """
        Args:
            aug_hidden_view1 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1
            aug_hidden_view2 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1

        Returns: nce_loss (FloatTensor, (,)): calculated nce loss
        """
        if aug_hidden_view1.ndim > 2:
            # flatten tensor
            aug_hidden_view1 = aug_hidden_view1.view(aug_hidden_view1.size(0), -1)
            aug_hidden_view2 = aug_hidden_view2.view(aug_hidden_view2.size(0), -1)

        if self.sim_type not in ['cos', 'dot']:
            raise Exception(f"Invalid similarity_type for cs loss: [current:{self.sim_type}]. "
                            f"Please choose from ['cos', 'dot']")

        if self.sim_type == 'cos':
            sim11 = self.cosinesim(aug_hidden_view1, aug_hidden_view1)
            sim22 = self.cosinesim(aug_hidden_view2, aug_hidden_view2)
            sim12 = self.cosinesim(aug_hidden_view1, aug_hidden_view2)
        elif self.sim_type == 'dot':
            # calc similarity
            sim11 = aug_hidden_view1 @ aug_hidden_view1.t()
            sim22 = aug_hidden_view2 @ aug_hidden_view2.t()
            sim12 = aug_hidden_view1 @ aug_hidden_view2.t()
        # mask non-calc value
        sim11[..., range(sim11.size(0)), range(sim11.size(0))] = float('-inf')
        sim22[..., range(sim22.size(0)), range(sim22.size(0))] = float('-inf')

        cl_logits1 = torch.cat([sim12, sim11], -1)
        cl_logits2 = torch.cat([sim22, sim12.t()], -1)
        cl_logits = torch.cat([cl_logits1, cl_logits2], 0) / self.temperature
        if mask is not None:
            cl_logits = torch.masked_fill(cl_logits, mask, float('-inf'))
        target = torch.arange(cl_logits.size(0)).long().to(aug_hidden_view1.device)
        cl_loss = self.criterion(cl_logits, target)

        return cl_loss

    def cosinesim(self, aug_hidden1, aug_hidden2):
        h = torch.matmul(aug_hidden1, aug_hidden2.T)
        h1_norm2 = aug_hidden1.pow(2).sum(dim=-1).sqrt().view(h.shape[0], 1)
        h2_norm2 = aug_hidden2.pow(2).sum(dim=-1).sqrt().view(1, h.shape[0])
        return h / (h1_norm2 @ h2_norm2)


class SelfContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.05):
        super(SelfContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size, batch_size, dtype=torch.bool).to(device)).float())
        self.eps = 1e-5

    def forward(self, q, k):
        q = F.normalize(q, dim=1)  # (bs, dim)  --->  (bs, dim)
        k = F.normalize(k, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([q, k], dim=0)  # (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1),
                                                representations.unsqueeze(0), dim=2)  # (2*bs, 2*bs)
        sim_qk = torch.diag(similarity_matrix, self.batch_size)  # (bs,)
        sim_kq = torch.diag(similarity_matrix, -self.batch_size)  # (bs,)

        nominator_qk = torch.exp(sim_qk / self.temperature)  # (bs,)
        negatives_qk = similarity_matrix[:self.batch_size, self.batch_size:]  # (bs, bs)
        denominator_qk = nominator_qk + torch.sum(self.negatives_mask * torch.exp(negatives_qk / self.temperature),
                                                  dim=1)

        nominator_kq = torch.exp(sim_kq / self.temperature)
        negatives_kq = similarity_matrix[self.batch_size:, :self.batch_size]
        denominator_kq = nominator_kq + torch.sum(self.negatives_mask * torch.exp(negatives_kq / self.temperature),
                                                  dim=1)

        loss_qk = torch.sum(-torch.log(nominator_qk / denominator_qk + self.eps)) / self.batch_size
        loss_kq = torch.sum(-torch.log(nominator_kq / denominator_kq + self.eps)) / self.batch_size
        loss = loss_qk + loss_kq

        return loss


class FullContrastiveLoss(nn.Module):
    def __init__(self, batch_size, num_r, num_nr, device='cuda', temperature=0.05):
        super(FullContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.num_r = num_r
        self.num_nr = num_nr

        self.register_buffer("temperature", torch.tensor(temperature).to(device))  # 超参数 温度
        self.register_buffer("rumor_mask", (~torch.eye(num_r, num_r, dtype=torch.bool).to(device)).float())
        self.register_buffer("nonrumor_mask", (~torch.eye(num_nr, num_nr, dtype=torch.bool).to(device)).float())
        self.eps = 1e-5

    def compute_loss(self, feature, label):
        """
        feature: (batch, dim)
        r: rumor nr: non-rumor
        """
        index_r = torch.nonzero(label).squeeze()
        index_nr = torch.nonzero(label == 0).squeeze()
        ft_r = torch.index_select(feature, dim=0, index=index_r)
        ft_nr = torch.index_select(feature, dim=0, index=index_nr)

        similarity_matrix_r = F.cosine_similarity(ft_r.unsqueeze(1), ft_r.unsqueeze(0), dim=2)
        similarity_matrix_nr = F.cosine_similarity(ft_nr.unsqueeze(1), ft_nr.unsqueeze(0), dim=2)
        similarity_matrix_r_nr = F.cosine_similarity(ft_r.unsqueeze(1), ft_nr.unsqueeze(0), dim=2)
        similarity_matrix_nr_r = F.cosine_similarity(ft_nr.unsqueeze(1), ft_r.unsqueeze(0), dim=2)

        nominator_r = torch.sum(self.rumor_mask * torch.exp(similarity_matrix_r / self.temperature), dim=1)
        nominator_nr = torch.sum(self.nonrumor_mask * torch.exp(similarity_matrix_nr / self.temperature), dim=1)

        denominator_r = nominator_r + torch.sum(torch.torch.exp(similarity_matrix_r_nr / self.temperature), dim=1)
        denominator_nr = nominator_nr + torch.sum(torch.torch.exp(similarity_matrix_nr_r / self.temperature), dim=1)

        loss_r = torch.sum(-torch.log(nominator_r / denominator_r + self.eps)) / self.num_r
        loss_nr = torch.sum(-torch.log(nominator_nr / denominator_nr + self.eps)) / self.num_nr
        loss = loss_r + loss_nr
        return loss

    def forward(self, text, image, label):
        text = F.normalize(text, dim=1)
        image = F.normalize(image, dim=1)

        loss = self.compute_loss(text, label) + self.compute_loss(image, label)

        return loss
