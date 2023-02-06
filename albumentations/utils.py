import inspect
import typing

import albumentations

__all__ = ["get_filtered_transforms", "get_dual_transforms", "get_image_only_transforms", "get_transforms"]


def get_filtered_transforms(
    base_classes: typing.Tuple[typing.Type, ...],
    custom_arguments: typing.Optional[typing.Dict[typing.Type, dict]] = None,
    except_augmentations: typing.Optional[typing.Set[typing.Type]] = None,
) -> typing.List[typing.Tuple[typing.Type, dict]]:
    custom_arguments = custom_arguments or {}
    except_augmentations = except_augmentations or set()

    result = []

    for name, cls in inspect.getmembers(albumentations):
        if not inspect.isclass(cls) or not issubclass(cls, (albumentations.BasicTransform, albumentations.BaseCompose)):
            continue

        if "DeprecationWarning" in inspect.getsource(cls) or "FutureWarning" in inspect.getsource(cls):
            continue

        if not issubclass(cls, base_classes) or any(cls == i for i in base_classes) or cls in except_augmentations:
            continue

        try:
            if issubclass(cls, albumentations.BasicIAATransform):
                continue
        except AttributeError:
            pass

        result.append((cls, custom_arguments.get(cls, {})))

    return result


def get_image_only_transforms(
    custom_arguments: typing.Optional[typing.Dict[typing.Type[albumentations.ImageOnlyTransform], dict]] = None,
    except_augmentations: typing.Optional[typing.Set[typing.Type[albumentations.ImageOnlyTransform]]] = None,
) -> typing.List[typing.Tuple[typing.Type, dict]]:
    return get_filtered_transforms((albumentations.ImageOnlyTransform,), custom_arguments, except_augmentations)


def get_dual_transforms(
    custom_arguments: typing.Optional[typing.Dict[typing.Type[albumentations.DualTransform], dict]] = None,
    except_augmentations: typing.Optional[typing.Set[typing.Type[albumentations.DualTransform]]] = None,
) -> typing.List[typing.Tuple[typing.Type, dict]]:
    return get_filtered_transforms((albumentations.DualTransform,), custom_arguments, except_augmentations)


def get_transforms(
    custom_arguments: typing.Optional[typing.Dict[typing.Type[albumentations.BasicTransform], dict]] = None,
    except_augmentations: typing.Optional[typing.Set[typing.Type[albumentations.BasicTransform]]] = None,
) -> typing.List[typing.Tuple[typing.Type, dict]]:
    return get_filtered_transforms(
        (albumentations.ImageOnlyTransform, albumentations.DualTransform), custom_arguments, except_augmentations
    )
