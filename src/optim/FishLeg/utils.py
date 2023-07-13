import re
import torch
import torch.nn as nn

from collections import namedtuple
from typing import List, Callable, Union

from .layers import *

from transformers.models.bert.modeling_bert import BertAttention

__all__ = [
    "recursive_setattr",
    "recursive_getattr",
    "update_dict",
    "_use_grad_for_differentiable",
    "initialise_FishModel",
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


NamedLayer = namedtuple("NamedLayerParam", ["layer_name", "layer"])


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
            if not any_str_or_regex_matches_param_name(
                layer_name, ["re:" + l for l in found_layer_names]
            ):
                named_layers.append(NamedLayer(layer_name, layer))
                found_layer_names.append(layer_name)
    if params_strict:
        validate_all_params_found(param_names, found_layer_names)

    return named_layers


def initialise_FishModel(
    model: nn.Module,
    module_names: str,
    fish_scale: int = 1.0,
    verbose: bool = False,
) -> Union[nn.Module, List]:
    """Given a model to optimize, parameters can be devided to

    #. those fixed as pre-trained.
    #. those required to optimize using FishLeg.

    Replace modules in the second group with FishLeg modules.

    Args:
        model (:class:`torch.nn.Module`, required):
            A model containing modules to replace with FishLeg modules
            containing extra functionality related to FishLeg algorithm.
    Returns:
        :class:`torch.nn.Module`, the replaced model.
    """
    if any(["weight" in m for m in module_names]):
        raise TypeError(
            f"Parameters to be optimized in FishLeg are considered together in one module, and cannot be optimized individually"
        )

    # Add auxiliary parameters
    if module_names == "__ALL__":
        named_layers = [
            NamedLayer(name, layer) for name, layer in model.named_modules()
        ]

    else:
        named_layers = get_named_layers_by_regex(model, module_names)

    replaced_layers = []
    for named_layer in named_layers:
        module = named_layer.layer
        module_name = named_layer.layer_name

        for layer in replaced_layers:
            if re.match(layer.layer_name, module_name):
                inner_name = module_name[len(layer.layer_name) + 1 :]
                if any(
                    re.match(inner_name, param_name) for param_name in layer.layer.order
                ):
                    continue

        if isinstance(module, nn.Linear):
            replace = FishLinear(
                module.in_features,
                module.out_features,
                module.bias is not None,
                init_scale=fish_scale,
                device=next(module.parameters()).device,
            )

            # if self.initialization == "normal":
            #     init.normal_(replace.weight, 0, 1 / np.sqrt(module.in_features))
            # elif self.initialization == "zero":  # fill with zeros for adapters
            #     module.weight.data.zero_()
            #     module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            replace = FishEmbedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                padding_idx=module.padding_idx,
                max_norm=module.max_norm,
                norm_type=module.norm_type,
                scale_grad_by_freq=module.scale_grad_by_freq,
                sparse=module.sparse,
                init_scale=fish_scale,
                device=next(module.parameters()).device,
            )

        elif isinstance(module, nn.Conv2d):
            replace = FishConv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode,
                init_scale=fish_scale,
                device=next(module.parameters()).device
                # TODO: deal with dtype and device?
            )

        elif isinstance(module, BertAttention):
            config = model.config
            replace = FishBertAttention(config, device=next(module.parameters()).device)

        elif isinstance(module, nn.BatchNorm2d):
            replace = FishBatchNorm2d(
                num_features=module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
                init_scale=fish_scale,
                device=next(module.parameters()).device,
            )

        elif isinstance(module, nn.LayerNorm):
            replace = FishLayerNorm(
                normalized_shape=module.normalized_shape,
                eps=module.eps,
                elementwise_affine=module.elementwise_affine,
                init_scale=fish_scale,
                device=next(module.parameters()).device,
            )
        else:
            continue

        replace = update_dict(replace, module)
        recursive_setattr(model, module_name, replace)
        replaced_layers.append(NamedLayer(module_name, replace))
        if verbose:
            print("Replaced with FishLeg Layer: \t ", module_name)

    return model


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
