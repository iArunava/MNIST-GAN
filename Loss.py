import torch

def real_loss(D_out, smooth=False, cuda=False):
    batch_size = D_out.shape[0]
    labels = torch.ones(batch_size)
    if cuda:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    batch_size = D_out.shape[0]
    labels = torch.zeros(batch_size)
    if cuda:
        labels = labels.cuda()
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss
