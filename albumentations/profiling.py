import time
from functools import wraps
from typing import Callable, Optional

from typing_extensions import ParamSpec

from .utils import get_transforms

P = ParamSpec("P")


class ProfileData:
    def __init__(self, name: str, dt: float, profile_data: Optional["ProfileData"]):
        self.name = name
        self.dt = dt
        self.profile_data = profile_data


def profile_wrapper(name: str) -> Callable:
    def _decorate(func: Callable[P, dict]) -> Callable[P, dict]:
        @wraps(func)
        def wrapped_func(*args: P.args, **kwargs: P.kwargs) -> dict:
            profile = kwargs.get("profile", False)
            if not profile:
                return func(*args, **kwargs)

            s = time.time()
            res = func(*args, **kwargs)
            dt = time.time() - s

            prof = res.get("profile_results", None)
            if prof is not None and not isinstance(prof, ProfileData):
                raise ValueError(f"Wrong type of `profile_results`. Must be {type(ProfileData)}. Got: {type(prof)}")

            res["profile_results"] = ProfileData(name=name, dt=dt, profile_data=prof)
            return res

        return wrapped_func

    return _decorate


def enable_profiling():
    transforms = get_transforms()

    wrapped_methods_names = (
        "__call__",
        "apply",
        "apply_to_bbox",
        "apply_to_bboxes",
        "apply_to_keypoint",
        "apply_to_keypoints",
        "apply_to_mask",
        "apply_to_masks",
    )
    for cls_obj, _ in transforms:
        cls_name = cls_obj.get_class_fullname()
        for name in wrapped_methods_names:
            if hasattr(cls_obj, name):
                setattr(cls_obj, name, profile_wrapper(f"{cls_name}.{name}")(getattr(cls_obj, name)))
