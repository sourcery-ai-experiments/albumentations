import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar, Union

from typing_extensions import ParamSpec

from .utils import BaseAndComposeType, get_transforms_with_compose

__all__ = ["Profiler"]

P = ParamSpec("P")
T = TypeVar("T")


class Profiler:
    @dataclass
    class Data:
        name: str
        dt: float
        profile_data: Optional["Profiler.Data"]

    def __init__(self):
        self._original_functions = self._get_functions()
        self._last_data: Optional[Profiler.Data] = None
        self._statistics = None

        # Set wrapper
        for name, (cls, func_name, func) in self._original_functions.items():
            setattr(cls, func_name, self._profile_wrapper(name, func))

    def __del__(self):
        # Remove profile wrapper
        for name, (cls, func_name, func) in self._original_functions.items():
            setattr(cls, func_name, func)

    @staticmethod
    def _get_functions() -> Dict[str, Tuple[BaseAndComposeType, str, Callable]]:
        transforms = get_transforms_with_compose()

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
        res = {}
        for cls_obj in transforms:
            cls_name = cls_obj.get_class_fullname()
            for name in wrapped_methods_names:
                if hasattr(cls_obj, name):
                    res[f"{cls_name}.{name}"] = cls_obj, name, getattr(cls_obj, name)
        return res

    def _profile_wrapper(self, name: str, func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapped_func(*args: P.args, **kwargs: P.kwargs) -> T:
            s = time.time()
            res = func(*args, **kwargs)
            dt = time.time() - s

            self._last_data = self.Data(name=name, dt=dt, profile_data=self._last_data)
            return res

        return wrapped_func

    def __enter__(self):
        if self._last_data is not None:
            raise RuntimeError

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError
