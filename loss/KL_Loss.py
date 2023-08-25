import torch.nn.functional as F

def compute_kl_loss(p, q, pad_mask=None, T=10):
    p_T = p / T
    q_T = q / T
    p_loss = F.kl_div(F.log_softmax(p_T, dim=-1), F.softmax(q_T, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q_T, dim=-1), F.softmax(p_T, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    # p_loss = p_loss.sum()
    # q_loss = q_loss.sum()

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss