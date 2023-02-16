# TODO Support correct call stack with sequences and composition transforms

import time
from functools import wraps
from types import TracebackType
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar

from typing_extensions import ParamSpec

from albumentations import BaseCompose
from .utils import BaseAndComposeType, get_transforms_with_compose

__all__ = ["Profiler"]

P = ParamSpec("P")
T = TypeVar("T")


_PROFILER_RUN_NAME = "Profiling"

_BBOX_METHOD = "apply_to_bbox"
_BBOXES_METHOD = "apply_to_bboxes"
_KEYPOINT_METHOD = "apply_to_keypoint"
_KEYPOINTS_METHOD = "apply_to_keypoints"
_IMAGE_METHOD = "apply"
_MASK_METHOD = "apply_to_mask"
_MASKS_METHOD = "apply_to_masks"
_ALL_TARGETS_CALL = "__call__"
_METHODS_TO_PROFILE = (
    _ALL_TARGETS_CALL,
    _MASK_METHOD,
    _BBOX_METHOD,
    _BBOXES_METHOD,
    _KEYPOINT_METHOD,
    _KEYPOINTS_METHOD,
    _MASK_METHOD,
    _MASKS_METHOD,
)


class ProfilerData:
    def __init__(
        self,
        cls: Optional[Type[BaseAndComposeType]],
        name: str,
        dt: float,
        profile_data: Optional["ProfilerData"] = None
    ):
        self.cls = cls
        self.name = name
        self.dt = dt
        self.profile_data = profile_data

        self._results = None
        self._total_time = float("nan")
        self._min_time = float("nan")
        self._max_time = float("nan")
        self._avg_time = float("nan")
        self._call_count = 0

    @property
    def results(self) -> dict:
        self._process_results()
        return self._results or {}

    # def __str__(self) -> str:
    #     self._process_results()
    #     return (
    #         f"{self.name}"
    #         f" Total: {self._total_time:.3f}"
    #         f" Min: {self._min_time:.3f}"
    #         f" Average: {self._avg_time:.3f}"
    #         f" Max: {self._max_time:.3f}"
    #     )

    def _process_results(self) -> None:
        if self._results is not None:
            return

        self._results = {}
        self._total_time = self.dt
        self._min_time = self.dt
        self._avg_time = self.dt
        self._max_time = self.dt

        by_obj = {}
        by_stack = {}
        sequence_call = []

        def add_func(data: ProfilerData, result: dict) -> None:
            if data.cls not in result:
                result[data.cls] = {}
            if data.name not in result[data.cls]:
                result[data.cls][data.name] = []
            result[data.cls][data.name].append(data)

        def process(data: Optional[ProfilerData], stack_data: dict) -> None:
            if data.profile_data is None:
                return

            add_func(data, by_obj)
            sequence_call.append(data)

            cur_stack_data = stack_data
            if issubclass(data.cls, BaseCompose):
                cur_stack_data = {}

            add_func(data, stack_data)
            process(data.profile_data, cur_stack_data)

        process(self.profile_data, by_stack)


class Profiler:
    def __init__(self):
        self._original_functions = self._get_functions()
        self._current_run: Optional[ProfilerData] = None
        self._last_run: Optional[ProfilerData] = None
        self._statistics = None
        self._started = False
        self._t_start = 0

        # Set wrapper
        for name, (cls, func_name, func) in self._original_functions.items():
            setattr(cls, func_name, self._profile_wrapper(cls, name, func))

    def __del__(self):
        # Remove profile wrapper
        for name, (cls, func_name, func) in self._original_functions.items():
            setattr(cls, func_name, func)

    @staticmethod
    def _get_functions() -> Dict[str, Tuple[BaseAndComposeType, str, Callable]]:
        transforms = get_transforms_with_compose()

        res = {}
        for cls_obj in transforms:
            cls_name = cls_obj.get_class_fullname()
            for name in _METHODS_TO_PROFILE:
                if hasattr(cls_obj, name):
                    res[f"{cls_name}.{name}"] = cls_obj, name, getattr(cls_obj, name)
        return res

    def _profile_wrapper(self, cls: Type[BaseAndComposeType], name: str, func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapped_func(*args: P.args, **kwargs: P.kwargs) -> T:
            if not self._started:
                return func(*args, **kwargs)

            s = time.time()
            res = func(*args, **kwargs)
            dt = time.time() - s

            self._current_run = ProfilerData(cls=cls, name=name, dt=dt, profile_data=self._current_run)
            return res

        return wrapped_func

    def start(self):
        if self._started:
            raise RuntimeError("Profiling already started.")
        if self._current_run is not None:
            raise RuntimeError("Current run data is not empty. Maybe you forget to call `stop`.")
        self._started = True
        self._t_start = time.time()

    def stop(self):
        if not self._started:
            raise RuntimeError("Profiler is not started. Call `start` first.")
        self._last_run = self._current_run
        self._current_run = None
        self._started = False
        self._last_run = ProfilerData(
            cls=None, name=_PROFILER_RUN_NAME, dt=time.time() - self._t_start, profile_data=self._last_run
        )

    def __enter__(self) -> "Profiler":
        self.start()
        return self

    def __exit__(self, exc_type: Type[Exception], exc_val: Exception, exc_tb: TracebackType) -> None:
        self.stop()

    def get_profile_results(self) -> Optional["ProfilerData"]:
        return self._last_run
