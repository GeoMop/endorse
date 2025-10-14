
from contextlib import ContextDecorator

class ReturnCode():
    UNKNOWN_ERROR = -1000
    BGEM_GEOM_ERROR = -1001
    BGEM_GMSH_ERROR = -1002
    BGEM_HEAL_ERROR = -1003
    ZARR_ERROR = -1100


class WrapperException(Exception):
    """Common wrapper Exception."""
    code: ReturnCode = ReturnCode.UNKNOWN_ERROR  # default for the class

    def __init__(self, msg: str | None = None, *, code: ReturnCode | None = None):
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

