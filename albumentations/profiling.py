# TODO Support correct call stack with sequences and composition transforms

import time
from functools import wraps
from types import TracebackType
from typing import Callable, Dict, Optional, Tuple, Type, TypeVar

from typing_extensions import ParamSpec

from .utils import BaseAndComposeType, get_transforms_with_compose

__all__ = ["Profiler"]

P = ParamSpec("P")
T = TypeVar("T")


_PROFILER_RUN_NAME = "Profiling"


class ProfilerData:
    def __init__(self, name: str, dt: float, profile_data: Optional["ProfilerData"] = None):
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

        def process(data: Optional[ProfilerData], result_data: dict) -> ProfilerData:
            data._total_time = data.dt
            data._min_time = data.dt
            data._avg_time = data.dt
            data._max_time = data.dt

            if data.profile_data is None:
                return data

            if data.name not in result_data:
                result_data[data.name] = []
            result_data[data.name].append(data)
            process(data.profile_data, result_data)

        process(self.profile_data, self._results)


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
            if not self._started:
                return func(*args, **kwargs)

            s = time.time()
            res = func(*args, **kwargs)
            dt = time.time() - s

            self._current_run = ProfilerData(name=name, dt=dt, profile_data=self._current_run)
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
        self._last_run = ProfilerData(_PROFILER_RUN_NAME, time.time() - self._t_start, self._last_run)

    def __enter__(self) -> "Profiler":
        self.start()
        return self

    def __exit__(self, exc_type: Type[Exception], exc_val: Exception, exc_tb: TracebackType) -> None:
        self.stop()

    def get_profile_results(self) -> Optional["ProfilerData"]:
        return self._last_run
