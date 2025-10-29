from enum import EnumMeta, IntEnum

class FirstEnumIsDefault(EnumMeta):
    default = object()
    def __call__(cls, value=default, *args, **kwargs):
        if value is FirstEnumIsDefault.default:
            # Assume the first enum is default
            return next(iter(cls))
        return super().__call__(value, *args, **kwargs)