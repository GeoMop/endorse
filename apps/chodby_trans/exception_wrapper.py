
from contextlib import ContextDecorator

class ReturnCode():
    OK = 0
    UNKNOWN_ERROR = -1000
    BGEM_GEOM_ERROR = -1001
    BGEM_GMSH_ERROR = -1002
    BGEM_HEAL_ERROR = -1003
    FLOW123_ERROR = -1010
    SAMPLE_ERROR = -1020
    FINE_TRANSPORT_ERROR = -1021
    COARSE_TRANSPORT_ERROR = -1022
    HOMOGENIZATION_ERROR = -1030
    ZARR_ERROR = -1100
    SKIP = -1999
    NONE = -2000

    @classmethod
    def to_list(cls):
        """Return a list of all return code values."""
        return sorted([
            value
            for name, value in vars(cls).items()
            if isinstance(value, int) and name != '__firstlineno__'
        ])

    @classmethod
    def to_dict(cls):
        """Return dict of {name: value}, sorted by value ascending."""
        items = [
            (name, value)
            for name, value in vars(cls).items()
            if isinstance(value, int) and name != '__firstlineno__'
        ]
        # Sort by the integer value
        items_sorted = sorted(items, key=lambda x: x[1])
        return dict(items_sorted)

    @classmethod
    def failed_list(cls):
        """Return all negative codes except the NONE code."""
        return [
            v
            for name, v in cls.to_dict().items()
            if v < 0 and name != "NONE"
        ]


class WrapperException(Exception):
    """Common wrapper Exception."""
    code: int = ReturnCode.UNKNOWN_ERROR

    def __init__(self, msg: str | None = None, *, code: int | None = None):
        if msg is None:
            # fall back to the docstring or a generic message
            msg = self.__class__.__doc__ or "Error"
        super().__init__(msg)
        # per-instance override; otherwise use the class default
        self.code = self.__class__.code if code is None else code

    def __repr__(self):
        return f"{self.__class__.__name__}(code={self.code!r}, msg={self.args[0]!r})"

class GeomException(WrapperException):
    """Errors originating from Gmsh geometry OCC model operations."""
    code = ReturnCode.BGEM_GEOM_ERROR

class MeshException(WrapperException):
    """Errors originating from Gmsh meshing operations."""
    code=ReturnCode.BGEM_GMSH_ERROR

class HealException(WrapperException):
    """Errors originating from Gmsh HealMesh operations."""
    code=ReturnCode.BGEM_HEAL_ERROR

class Flow123dException(WrapperException):
    """Errors originating from running Flow123d."""
    code=ReturnCode.FLOW123_ERROR


class FineTransportException(WrapperException):
    """Failure in the fine-model stage of a transport pair."""

    code = ReturnCode.FINE_TRANSPORT_ERROR


class CoarseTransportException(WrapperException):
    """Failure in the coarse-model stage after a completed fine stage."""

    code = ReturnCode.COARSE_TRANSPORT_ERROR

    def __init__(
        self,
        msg: str | None = None,
        *,
        code: int | None = None,
        fine_return_code: int = ReturnCode.NONE,
        fine_values: object | None = None,
        fine_eval_time: float = -1.0,
    ):
        super().__init__(msg, code=code)
        self.fine_return_code = int(fine_return_code)
        self.fine_values = fine_values
        self.fine_eval_time = float(fine_eval_time)


class HomogenizationException(WrapperException):
    """Failure while constructing the homogenized coarse conductivity."""

    code = ReturnCode.HOMOGENIZATION_ERROR



class wrap_as(ContextDecorator):
    """
    Context manager/decorator that rethrows any Exception as `exc_cls`,
    preserving the original traceback via exception chaining.

    Example usage:
        import my_exceptions as MyExceptions
        from exception_wrapper import wrap_as, rethrow_as

        # Use as a decorator
        @rethrow_as(MyExceptions.GmshException, "Running Gmsh")
        def build_mesh():
            # ... risky code ...
            raise ValueError("invalid option")  # example

        # Use as a context manager
        def solve():
            with wrap_as(MyExceptions.SolverException, "Solving system"):
                # ... risky code ...
                raise RuntimeError("matrix is singular")  # example
    """
    def __init__(self, exc_cls, msg=None):
        self.exc_cls = exc_cls
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            return False  # nothing to do
        if isinstance(exc, self.exc_cls):
            return False  # already the right type; let it propagate
        message = f"{self.msg}: {exc}" if self.msg else str(exc)
        raise self.exc_cls(message) from exc


def rethrow_as(exc_cls, msg=None):
    """Decorator form of `wrap_as` for functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with wrap_as(exc_cls, msg):
                return func(*args, **kwargs)
        return wrapper
    return decorator

