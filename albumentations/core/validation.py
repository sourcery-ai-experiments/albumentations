from inspect import Parameter, signature
from typing import Any, Callable, Dict, Optional, Tuple, Type

from pydantic import BaseModel


class ValidatedTransformMeta(type):
    def __new__(cls: Type[Any], name: str, bases: Tuple[type, ...], dct: Dict[str, Any]) -> Type[Any]:
        if "InitSchema" in dct and issubclass(dct["InitSchema"], BaseModel):
            original_init: Optional[Callable[..., Any]] = dct.get("__init__")
            if original_init is None:
                msg = "__init__ not found in class definition"
                raise ValueError(msg)

            def custom_init(self: Any, *args: Any, **kwargs: Any) -> None:
                init_params = signature(original_init).parameters
                param_names = list(init_params.keys())[1:]  # Exclude 'self'
                full_kwargs: Dict[str, Any] = dict(zip(param_names, args))
                full_kwargs.update(kwargs)

                for name, param in init_params.items():
                    if name != "self" and name not in full_kwargs and param.default is not Parameter.empty:
                        full_kwargs[name] = param.default

                # No try-except block needed as we want the exception to propagate naturally
                config = dct["InitSchema"](**full_kwargs)

                validated_kwargs = config.dict()
                original_init(self, **validated_kwargs)

            # Rename __init__ to custom_init to avoid the N807 warning
            dct["__init__"] = custom_init

        return super().__new__(cls, name, bases, dct)