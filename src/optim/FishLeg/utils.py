import torch.nn as nn
import torch
from collections import OrderedDict, namedtuple
from typing import List
import torch.nn as nn
import regex as re

__all__ = [
    "recursive_setattr",
    "recursive_getattr",
    "update_dict",
    "_use_grad_for_differentiable",
    "get_named_layers_by_regex"
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


def _use_grad_for_differentiable(func):
    def _use_grad(self, *args, **kwargs):
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults["differentiable"])
            ret = func(self, *args, **kwargs)
        finally:
            torch.set_grad_enabled(prev_grad)
        return ret

    return _use_grad

def any_str_or_regex_matches_param_name(
    param_name: str,
    name_or_regex_patterns: List[str],
) -> bool:
    """
    :param param_name: The name of a parameter
    :param name_or_regex_patterns: List of full param names to match to the input or
        regex patterns to match with that should be prefixed with 're:'
    :return: True if any given str or regex pattern matches the given name
    """
    for name_or_regex in name_or_regex_patterns:
        if name_or_regex[:3] == "re:":
            pattern = name_or_regex[3:]
            if re.match(pattern, param_name):
                return True
        else:
            if param_name == name_or_regex:
                return True
    return False


def validate_all_params_found(
    name_or_regex_patterns: List[str],
    found_param_names: List[str],
):
    """
    :param name_or_regex_patterns: List of full param names or regex patterns of them
        to check for matches in found_param_names names
    :param found_param_names: List of NamedLayerParam objects to check for matches
    :raise RuntimeError: If there is a name or regex pattern that does not have a
        match in found_param_names
    """
    for name_or_regex in name_or_regex_patterns:
        if "re:" != name_or_regex[:3] and name_or_regex in found_param_names:
            continue  # name found in list of full parameter names
        if "re:" == name_or_regex[:3] and any(
            re.match(name_or_regex[3:], name) for name in found_param_names
        ):
            continue  # regex pattern matches at least one full parameter name

        raise RuntimeError(
            "All supplied parameter names or regex patterns not found."
            "No match for {} in found parameters {}. \nSupplied {}".format(
                name_or_regex, found_param_names, name_or_regex_patterns
            )
        )

NamedLayer = namedtuple(
    "NamedLayerParam", ["layer_name", "layer"]
)

def get_named_layers_by_regex(
    module: nn.Module,
    param_names: List[str],
    params_strict: bool = False,
) -> List[NamedLayer]:
    """
    :param module: the module to get the matching layers and params from
    :param param_names: a list of names or regex patterns to match with full parameter
        paths. Regex patterns must be specified with the prefix 're:'
    :param params_strict: if True, this function will raise an exception if there a
        parameter is not found to match every name or regex in param_names
    :return: a list of NamedLayerParam tuples whose full parameter names in the given
        module match one of the given regex patterns or parameter names
    """
    named_layers = []
    found_layer_names = []
    for layer_name, layer in module.named_modules():
        if any_str_or_regex_matches_param_name(layer_name, param_names):
            if not any_str_or_regex_matches_param_name(layer_name, 
                                                        ['re:' + l for l in found_layer_names]):
                named_layers.append(
                    NamedLayer(layer_name, layer)
                )
                found_layer_names.append(layer_name)
    if params_strict:
        validate_all_params_found(param_names, found_layer_names)

    return named_layers


