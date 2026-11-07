"""Guarded optional import of undecomposed Stim detector error models.

Semantics: https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md
Each error instruction remains one independent mechanism, including hyperedges
and logical-only errors. This adapter never calls unbounded DEM.flattened().
"""
from math import isfinite

from qeclang.algebra.binary import nonnegative_integer
from .detector_model import DetectorModel, _limits


def import_detector_model(dem, *, gauge_free=False, num_detectors=None, num_observables=None,
                  max_faults=1_000_000, max_detectors=1_000_000,
                  max_observables=1_000_000, max_nonzeros=10_000_000,
                  max_expanded_instructions=1_000_000,
                  max_expanded_targets=10_000_000, max_nesting=64):
    """Import with preflight bounds before expanding repeats or allocating rows.

    The caller must establish gauge-free provenance, normally by obtaining the
    DEM with circuit.detector_error_model(allow_gauge_detectors=False,
    decompose_errors=False, approximate_disjoint_errors=False). Raw DEM data
    cannot reveal whether an error(0.5) came from gauge randomness; gauge_free
    is an explicit caller assertion, not a property inferred from probabilities.

    Expanded instruction work includes repeat-body visits, even empty bodies;
    expanded target work includes arguments and repeated IDs before cancellation.
    Nesting is limited to at most 128 levels. Coordinates/tags are annotations
    and are not retained. Separator decompositions and unknown operations fail.
    Optional dimensions may extend trailing isolated nodes, never shrink them.
    Stim targets_copy/args_copy allocate foreign per-instruction lists before
    their lengths can be counted; the limits are not caps on that input-copy
    memory or on an already constructed Stim object.
    """
    try:
        import stim
    except ImportError as exc:
        raise ImportError("Stim DEM import requires qeclang[circuits]") from exc
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError("DEM import requires a stim.DetectorErrorModel")
    if gauge_free is not True:
        raise ValueError("gauge-free DEM provenance must be established explicitly")
    nf, nd, no, nnz = _limits(max_faults, max_detectors, max_observables, max_nonzeros)
    instructions = nonnegative_integer(max_expanded_instructions, "max_expanded_instructions")
    targets_limit = nonnegative_integer(max_expanded_targets, "max_expanded_targets")
    nesting = nonnegative_integer(max_nesting, "max_nesting")
    if nesting > 128:
        raise ValueError("max_nesting must not exceed 128")

    def check(work, targets, faults, detector, observable):
        if work > instructions:
            raise ValueError("DEM expansion exceeds max_expanded_instructions")
        if targets > targets_limit:
            raise ValueError("DEM expansion exceeds max_expanded_targets")
        if faults > nf:
            raise ValueError("DEM expansion exceeds max_faults")
        if detector >= nd or observable >= no:
            raise ValueError("DEM expansion exceeds detector/observable axis limits")

    def parse(block, depth):
        if depth > nesting:
            raise ValueError("DEM repeat nesting exceeds max_nesting")
        records = []
        offset = work = targets = faults = 0
        largest_detector = largest_observable = -1
        for instruction in block:
            if isinstance(instruction, stim.DemRepeatBlock):
                repeat = int(instruction.repeat_count)
                child, shift, child_work, child_targets, child_faults, child_d, child_l = parse(
                    instruction.body_copy(), depth + 1)
                work += repeat * (child_work + 1)
                targets += repeat * child_targets
                faults += repeat * child_faults
                if child_d >= 0:
                    largest_detector = max(largest_detector, offset + (repeat-1)*shift + child_d)
                largest_observable = max(largest_observable, child_l)
                offset += repeat*shift
                records.append(("repeat", repeat, child))
            else:
                kind = instruction.type
                args = tuple(instruction.args_copy())
                items = tuple(instruction.targets_copy())
                if any(not isfinite(a) for a in args):
                    raise ValueError("DEM instruction arguments must be finite")
                work += 1
                targets += len(items) + len(args)
                if kind == "shift_detectors":
                    if len(items) != 1 or type(items[0]) is not int or items[0] < 0:
                        raise ValueError("invalid detector shift")
                    offset += items[0]
                    records.append(("shift", items[0]))
                elif kind in ("error", "detector", "logical_observable"):
                    ds, ls = [], []
                    for target in items:
                        if not isinstance(target, stim.DemTarget) or target.is_separator():
                            raise ValueError("separator decompositions or unsupported DEM targets are not supported")
                        if target.is_relative_detector_id():
                            if kind == "logical_observable":
                                raise ValueError("invalid logical observable declaration")
                            ds.append(int(target.val))
                            largest_detector = max(largest_detector, offset + target.val)
                        elif target.is_logical_observable_id():
                            if kind == "detector":
                                raise ValueError("invalid detector declaration")
                            ls.append(int(target.val))
                            largest_observable = max(largest_observable, target.val)
                        else:
                            raise ValueError("unsupported DEM target")
                    if kind == "error":
                        if len(args) != 1 or not 0 <= args[0] <= 1:
                            raise ValueError("invalid DEM error probability")
                        faults += 1
                        records.append(("error", args[0], tuple(ds), tuple(ls)))
                    elif kind == "logical_observable" and args:
                        raise ValueError("unsupported logical observable arguments")
                else:
                    raise ValueError(f"unsupported DEM instruction: {kind}")
            check(work, targets, faults, largest_detector, largest_observable)
        return tuple(records), offset, work, targets, faults, largest_detector, largest_observable

    records, _, _, _, _, largest_d, largest_l = parse(dem, 0)
    detectors = largest_d+1 if num_detectors is None else nonnegative_integer(num_detectors, "num_detectors")
    observables = largest_l+1 if num_observables is None else nonnegative_integer(num_observables, "num_observables")
    if detectors <= largest_d or observables <= largest_l:
        raise ValueError("explicit dimensions cannot omit DEM targets")
    if detectors > nd or observables > no:
        raise ValueError("explicit DEM dimensions exceed axis limits")

    def expand(block, offset):
        for record in block:
            if record[0] == "shift":
                offset += record[1]
            elif record[0] == "repeat":
                for _ in range(record[1]):
                    offset = yield from expand(record[2], offset)
            else:
                _, probability, ds, ls = record
                yield probability, tuple(offset+d for d in ds), ls
        return offset

    return DetectorModel.from_errors(expand(records, 0), num_detectors=detectors,
                                      num_observables=observables, max_faults=nf,
                                      max_detectors=nd, max_observables=no, max_nonzeros=nnz)


from_stim_dem = import_detector_model

__all__ = ["import_detector_model", "from_stim_dem"]
