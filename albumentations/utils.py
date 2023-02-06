import inspect
from typing import Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import albumentations as A

__all__ = [
    "get_filtered_transforms",
    "get_dual_transforms",
    "get_image_only_transforms",
    "get_transforms",
    "get_transforms_with_compose",
]

BaseTransformType = Type[A.BasicTransform]
BaseAndComposeType = Union[Type[A.BaseCompose], Type[A.BasicTransform]]


def get_filtered_transforms(
    base_classes: Tuple[BaseAndComposeType, ...],
    custom_arguments: Optional[Dict[Type, dict]] = None,
    except_augmentations: Optional[Set[Type]] = None,
) -> List[Tuple[BaseAndComposeType, dict]]:
    custom_arguments = custom_arguments or {}
    except_augmentations = except_augmentations or set()

    result = []

    for name, cls in inspect.getmembers(A):
        if not inspect.isclass(cls) or not issubclass(cls, (A.BasicTransform, A.BaseCompose)):
            continue

        if "DeprecationWarning" in inspect.getsource(cls) or "FutureWarning" in inspect.getsource(cls):
            continue

        if not issubclass(cls, base_classes) or any(cls == i for i in base_classes) or cls in except_augmentations:
            continue

        try:
            if issubclass(cls, A.BasicIAATransform):
                continue
        except AttributeError:
            pass

        result.append((cls, custom_arguments.get(cls, {})))

    return result


def get_image_only_transforms(
    custom_arguments: Optional[Dict[Type[A.ImageOnlyTransform], dict]] = None,
    except_augmentations: Optional[Set[Type[A.ImageOnlyTransform]]] = None,
) -> List[Tuple[BaseTransformType, dict]]:
    return get_filtered_transforms((A.ImageOnlyTransform,), custom_arguments, except_augmentations)  # type: ignore


def get_dual_transforms(
    custom_arguments: Optional[Dict[Type[A.DualTransform], dict]] = None,
    except_augmentations: Optional[Set[Type[A.DualTransform]]] = None,
) -> List[Tuple[BaseTransformType, dict]]:
    return get_filtered_transforms((A.DualTransform,), custom_arguments, except_augmentations)  # type: ignore


def get_transforms(
    custom_arguments: Optional[Dict[Type[A.BasicTransform], dict]] = None,
    except_augmentations: Optional[Set[Type[A.BasicTransform]]] = None,
) -> List[Tuple[BaseTransformType, dict]]:
    return get_filtered_transforms(
        (A.ImageOnlyTransform, A.DualTransform), custom_arguments, except_augmentations
    )  # type: ignore


def get_transforms_with_compose() -> List[BaseAndComposeType]:
    return [i[0] for i in get_filtered_transforms((A.ImageOnlyTransform, A.DualTransform, A.BaseCompose), None, None)]
