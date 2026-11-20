"""Conditional interval transport along exact, sector-preserving CSS metrics.

These functions prove an implication, not the source interval premise. A raw
distance_claim dictionary is self-asserted data. Workflow execution must obtain
it from a checked predecessor; an independent inspector must require its digest
in its own checked-interval set. No function here labels a premise verified.
"""
from qeclang.algebra.binary import nonnegative_integer
from qeclang.analysis.certificate_io import presentation_fingerprint
from qeclang.codes import CSSCode
from .low_weight_payload import distance_claim_payload
from .relations import RelationError, check_relation
from .row_equivalence import row_equivalence_work, verify_row_equivalence


def _presentations(source, target, max_axis):
    axis = nonnegative_integer(max_axis, "max_axis")
    if not isinstance(source, CSSCode) or not isinstance(target, CSSCode):
        raise RelationError("distance transport requires positive packed CSS presentations")
    if source.n != target.n:
        raise RelationError("distance transport requires equal physical coordinate counts")
    if max(source.n, source.hx.nrows, source.hz.nrows, target.hx.nrows, target.hz.nrows) > axis:
        raise RelationError("distance transport exceeds max_axis")


def _parameters(params):
    if type(params) is not dict or len(params) != 2:
        raise RelationError("distance transport requires explicit relation parameters")
    relation = params.get("relation")
    if relation == "permutation":
        if set(params) != {"relation", "permutation"} or type(params["permutation"]) is not list:
            raise RelationError("permutation transport requires a JSON permutation array")
    elif relation == "generator_spaces":
        if set(params) != {"relation", "maps"} or type(params["maps"]) is not dict:
            raise RelationError("generator-space transport requires explicit bidirectional maps")
    else:
        raise RelationError("unsupported distance-preserving relation")
    return relation


def transport_interval_work(source, target, params, *, max_axis=100_000):
    """Proxy for relation checking, excluding hashing and arbitrary-precision width.

    Generator-space transport uses row_equivalence_work unchanged. Permutation
    transport counts physical-index visits, matrix-row visits and support visits.
    This estimator checks basic shapes, not validity of the relation or premise.
    Runtime callers should reserve these units before either relation checker.
    """
    _presentations(source, target, max_axis)
    relation = _parameters(params)
    if relation == "generator_spaces":
        return row_equivalence_work(source, target, params["maps"])
    if len(params["permutation"]) != source.n:
        raise RelationError("permutation must contain every physical site")
    rows = source.hx.nrows + source.hz.nrows + target.hx.nrows + target.hz.nrows
    old_support = source.hx.nnz + source.hz.nnz
    new_support = target.hx.nnz + target.hz.nnz
    return 8 + 4 * source.n + 4 * rows + 2 * old_support + new_support


def _metric_relation(source, target, params, max_axis, max_work):
    _presentations(source, target, max_axis)
    relation = _parameters(params)
    if max_work is not None:
        limit = nonnegative_integer(max_work, "max_work")
        if transport_interval_work(source, target, params, max_axis=max_axis) > limit:
            raise RelationError("distance transport exceeds max_work")
    if relation == "permutation":
        return check_relation("css.permute", source, target, {"permutation": params["permutation"]})
    return verify_row_equivalence(source, target, params["maps"])


def _interval(code, claim):
    try:
        distance_claim_payload(claim)
    except (ValueError, TypeError) as error:
        raise RelationError("malformed finite distance interval") from error
    if claim["source"] != presentation_fingerprint(code):
        raise RelationError("distance interval belongs to a different physical presentation")
    if claim["upper_bound"] > code.n:
        raise RelationError("finite witness-based interval exceeds the physical code width")
    return claim


def transport_interval_conclusion(source, target, claim, params, *, max_axis=100_000, max_work=None):
    """Return the conditional target conclusion under a checked-source premise.

    The source interval must already be established by the caller's scientific
    evidence path. This function validates its syntax/binding and the exact
    metric relation, but does not authenticate the lower/upper premise itself.
    Sector (x/z/pauli), finite endpoints and exact/bounded status stay unchanged.
    Logical existence belongs to the caller's checked premise; k and distance
    are not computed here. Decoder/circuit performance is outside this relation.
    """
    _presentations(source, target, max_axis)
    _interval(source, claim)
    _metric_relation(source, target, params, max_axis, max_work)
    return {**claim, "source": presentation_fingerprint(target)}


def check_transported_interval(source, target, source_claim, target_claim, params, *,
                               max_axis=100_000, max_work=None):
    """Check the implication edge, independently of the conclusion producer.

    Returning None establishes equal metric relations and equal interval fields
    with the appropriate source/target binding. It does not verify a self-asserted
    source claim. The caller must separately possess a checked source premise.
    No rank, solve, search, certificate producer, or interval producer is called.
    """
    _presentations(source, target, max_axis)
    _interval(source, source_claim)
    _interval(target, target_claim)
    _metric_relation(source, target, params, max_axis, max_work)
    for name in ("schema", "sector", "lower_bound", "upper_bound", "status"):
        if source_claim[name] != target_claim[name]:
            raise RelationError(f"transported distance interval changed {name}")


__all__ = ["transport_interval_conclusion", "check_transported_interval", "transport_interval_work"]
