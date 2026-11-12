"""Configuration graph copying with explicit pandas ownership.

Deep-copies DataFrames, Series, Index objects, cell values, and callable curves
under a single shared memo dictionary so each :class:`GridCase` owns an isolated
snapshot without breaking internal object references.  Uses public APIs only;
extension objects retain the standard ``__deepcopy__`` protocol.
"""

from copy import copy, deepcopy
from dis import get_instructions
from functools import partial
from types import (CellType, CodeType, FunctionType, MappingProxyType,
                   MethodType, ModuleType)
from typing import Any, Dict, cast

import numpy as np
import pandas as pd


def _function_globals(function: FunctionType) -> Dict[str, Any]:
    """Globals actually loaded by the code, including nested function bodies.

    Modules remain shared numerical dependencies.  Dynamic globals()/external
    module-state access is outside the configuration snapshot contract.
    """
    names: set[str] = set()

    def collect(code: CodeType) -> None:
        names.update(instruction.argval for instruction in get_instructions(code)
                     if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"))
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                collect(constant)

    collect(function.__code__)
    return {name: function.__globals__[name] for name in sorted(names)
            if name in function.__globals__ and name != "__builtins__"}


def _slot_names(value: Any) -> list[str]:
    """Declared instance state of a callable curve without a __dict__."""
    names = []
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        for name in ((slots,) if isinstance(slots, str) else slots):
            if name in ("__dict__", "__weakref__"):
                continue
            if name.startswith("__") and not name.endswith("__"):
                name = "_" + cls.__name__.lstrip("_") + name
            if name not in names:
                names.append(name)
    return names


def _callable_state(value: Any) -> Dict[str, Any]:
    """Curve parameters, respecting pandapower's declared JSON exclusions.

    Excludes ``json_excludes`` attributes on ``pandapower`` ``JSONSerializableClass``
    instances (such as lazy interpolators on ``SplineCharacteristic``) so curve
    evaluation does not change the configuration digest.  Other extension objects
    retain all attributes.
    """
    state = getattr(value, "__dict__", {})
    pandapower_curve = any(cls.__module__ == "pandapower.io_utils"
                           and cls.__name__ == "JSONSerializableClass"
                           for cls in type(value).__mro__)
    excludes = getattr(value, "json_excludes", ()) if pandapower_curve else ()
    return {name: child for name, child in state.items() if name not in excludes}


def copy_configuration(value: Any) -> Any:
    """Copy tables/cells/axes, preserving aliases and cycles inside the copy.

    Uses a single per-call memo dictionary so internal back-references (e.g. a
    controller referencing ``net``) resolve to the copied network.  Custom
    extension copy hooks are passed the shared deepcopy memo.
    """
    memo: Dict[int, Any] = {}

    def clone(source: Any) -> Any:
        result: Any
        identity = id(source)
        if identity in memo:
            return memo[identity]
        if isinstance(source, ModuleType) or source is pd.NA or source is pd.NaT:
            return source
        if (type(source).__name__ == "_ArrayFunctionDispatcher"
                and type(source).__module__.startswith("numpy")):
            # NumPy 2's importable array-function wrappers are immutable
            # numerical dependencies (copy/deepcopy return the same wrapper).
            return source
        if isinstance(source, FunctionType):
            # deepcopy treats functions as immutable, but their defaults,
            # closure cells and directly referenced parameters may be mutable.
            # Install the new function before cloning captures to retain cycles.
            namespace = {"__builtins__": source.__globals__.get("__builtins__", __builtins__)}
            # Preserve __name__, __package__, and __spec__ for relative imports
            # while cloning only globals referenced by bytecode.
            namespace.update({name: source.__globals__[name]
                              for name in ("__name__", "__package__", "__spec__")
                              if name in source.__globals__})
            captured = tuple(CellType() for _ in source.__closure__) if source.__closure__ else None
            result = FunctionType(source.__code__, namespace, source.__name__, None, captured)
            memo[identity] = result
            namespace.update({name: clone(child) for name, child in _function_globals(source).items()})
            result.__defaults__ = clone(source.__defaults__)
            result.__kwdefaults__ = clone(source.__kwdefaults__)
            result.__annotations__ = clone(source.__annotations__)
            result.__dict__.update(clone(vars(source)))
            result.__module__, result.__qualname__ = source.__module__, source.__qualname__
            for original, owned in zip(source.__closure__ or (), captured or ()):
                try:
                    child = original.cell_contents
                except ValueError:       # preserve an empty closure cell
                    continue
                owned.cell_contents = clone(child)
            return result
        if isinstance(source, MethodType):
            result = MethodType(clone(source.__func__), clone(source.__self__))
            memo[identity] = result
            return result
        if isinstance(source, partial):
            result = partial(source.func)
            memo[identity] = result
            result.__setstate__((clone(source.func), clone(source.args),
                                 clone(source.keywords), clone(vars(source))))
            return result
        if isinstance(source, pd.Index):
            if isinstance(source, pd.RangeIndex):
                # Even deep copies can share RangeIndex's cached ndarray.
                result = pd.RangeIndex(source.start, source.stop, source.step, name=source.name)
            elif isinstance(source, pd.CategoricalIndex):
                cat_codes: Any = source.codes.copy()
                categories = pd.Categorical.from_codes(cat_codes,
                                                       categories=clone(source.categories),
                                                       ordered=cast(pd.CategoricalDtype, source.dtype).ordered)
                result = pd.CategoricalIndex(categories, name=source.name)
            elif isinstance(source, pd.MultiIndex):
                # MultiIndex.copy also retains a previously materialised tuple
                # array. Rebuild through public levels/codes to discard it.
                level_codes: Any = [code.copy() for code in source.codes]
                result = pd.MultiIndex(levels=[clone(level) for level in source.levels],
                                       codes=level_codes,
                                       names=clone(source.names), sortorder=getattr(source, "sortorder", None),
                                       verify_integrity=False)
            else:
                result = source.copy(deep=True)
            memo[identity] = result
            return result
        if isinstance(source, (pd.DataFrame, pd.Series)):
            # DataFrame.copy eagerly deep-copies attrs with its own memo and
            # recurses forever if attrs point back to the frame. A constructor
            # copies storage without that finalisation step; clone attrs below.
            if isinstance(source, pd.DataFrame):
                # Allocate a fresh object-dtype frame instead of passing source
                # directly, avoiding shared BlockManager references and weakrefs
                # back to source. Object boxing keeps mixed int64/float columns
                # exact (no float64 interleaving).
                result = type(source)(np.empty(source.shape, dtype=object),
                                      index=clone(source.index), columns=clone(source.columns))
            else:
                dtype = clone(source.dtype)
                result = type(source)(source.to_numpy(copy=True), dtype=dtype,
                                      index=clone(source.index), name=source.name)
            memo[identity] = result
            if isinstance(source, pd.DataFrame):
                values = source.to_numpy(dtype=object, copy=False)
                for column, dtype in enumerate(source.dtypes):
                    if dtype == np.dtype(object):
                        cells = np.empty(len(source), dtype=object)
                        for row in range(len(source)):
                            cells[row] = clone(values[row, column])
                        result.isetitem(column, cells)
                    else:
                        result.isetitem(column, pd.array(values[:, column], dtype=clone(dtype), copy=True))
            elif source.dtype == np.dtype(object):
                cells = np.empty(len(source), dtype=object)
                for row, cell in enumerate(source.to_numpy(copy=False)):
                    cells[row] = clone(cell)
                result.iloc[:] = cells
            result.attrs = clone(source.attrs)
            result.flags.allows_duplicate_labels = source.flags.allows_duplicate_labels
            return result
        if isinstance(source, pd.CategoricalDtype):
            result = pd.CategoricalDtype(categories=clone(source.categories), ordered=source.ordered)
            memo[identity] = result
            return result
        if isinstance(source, dict):
            # copy() preserves pandapowerNet/defaultdict types and their public
            # mapping behaviour; avoid their table-specific deepcopy shortcuts.
            result = copy(source)
            if result is source:
                raise ValueError("a configuration mapping's copy protocol returned the source")
            memo[identity] = result
            result.clear()
            for key, item in source.items():
                result[clone(key)] = clone(item)
            if hasattr(source, "__dict__"):
                result.__dict__.update(clone(vars(source)))
            return result
        if isinstance(source, list):
            result = []
            memo[identity] = result
            result.extend(clone(item) for item in source)
            return result
        if isinstance(source, tuple):
            items = [clone(item) for item in source]
            # A tuple -> list -> tuple cycle may already have installed the
            # tuple while recursively copying its mutable child.
            if identity in memo:
                return memo[identity]
            result = type(source)(*items) if hasattr(source, "_fields") else tuple(items)
            memo[identity] = result
            return result
        if isinstance(source, set):
            result = set()
            memo[identity] = result
            result.update(clone(item) for item in source)
            return result
        if isinstance(source, frozenset):
            items_fs = [clone(item) for item in source]
            if identity in memo:
                return memo[identity]
            result = type(source)(items_fs)
            memo[identity] = result
            return result
        if isinstance(source, MappingProxyType):
            backing: Dict[Any, Any] = {}
            result = MappingProxyType(backing)
            memo[identity] = result
            for key, item in source.items():
                backing[clone(key)] = clone(item)
            return result
        if isinstance(source, np.ndarray) and source.dtype.hasobject:
            result = source.copy()
            memo[identity] = result
            for position in np.ndindex(source.shape):
                result[position] = clone(source[position])
            return result
        slots = _slot_names(source) if callable(source) and not isinstance(source, type) else []
        if ((hasattr(source, "__dict__") and not isinstance(source, type) or slots)
                and not hasattr(type(source), "__deepcopy__")):
            # Plain extension/controller objects may themselves own tables.
            # A normal deepcopy would re-enter pandas' shallow-cell protocol.
            # Explicit custom deepcopy hooks, when present, retain authority.
            result = copy(source)
            if result is source:
                raise ValueError("an extension copy protocol returned the mutable source")
            memo[identity] = result
            for name, child in getattr(source, "__dict__", {}).items():
                vars(result)[name] = clone(child)
            for name in slots:
                if hasattr(source, name):
                    object.__setattr__(result, name, clone(getattr(source, name)))
            return result
        return deepcopy(source, memo)

    return clone(value)
