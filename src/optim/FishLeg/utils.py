import torch.nn as nn

__all__ = [
    "recursive_setattr",
    "recursive_getattr",
    "update_dict",
]


def recursive_setattr(obj, attr, value):
    attr = attr.split(".", 1)

    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)


def recursive_getattr(obj, attr):
    attr = attr.split(".", 1)

    if len(attr) == 1:
        return getattr(obj, attr[0])
    else:
        return recursive_getattr(getattr(obj, attr[0]), attr[1])


def update_dict(replace: nn.Module, module: nn.Module) -> nn.Module:
    replace_dict = replace.state_dict()
    pretrained_dict = {
        k: v for k, v in module.state_dict().items() if k in replace_dict
    }
    replace_dict.update(pretrained_dict)
    replace.load_state_dict(replace_dict)
    return replace
