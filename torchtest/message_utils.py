_is_verbose = False


def verbose_on():
    global _is_verbose
    _is_verbose = True


def verbose_off():
    global _is_verbose
    _is_verbose = False


def is_verbose():
    return _is_verbose


def make_message(error_items, tensor):
    message = " ".join([_ for _ in error_string if _ is not None])
    if is_verbose():
        message += f"\nThe tensor is:\n{tensor}"
    return message
