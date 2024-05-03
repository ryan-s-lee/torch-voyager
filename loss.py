from torch import nn


def get_loss(loss_type):
    if loss_type == "multilabel":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "globalaccess":
        return nn.CrossEntropyLoss()
    else:
        raise RuntimeError("Invalid loss type")
