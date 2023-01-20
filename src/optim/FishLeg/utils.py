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
