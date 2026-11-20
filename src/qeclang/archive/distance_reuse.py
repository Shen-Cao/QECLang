"""Reconstruct native checked distance premises for explicit cross-run transport.

Archive reads are freshly inspected. Assembly checks data and provenance
bindings; the supplied metric relation is checked by mandatory execution or
inspection of the returned program. No search, elimination, or write occurs.
"""
from qeclang.analysis.certificate_io import presentation_fingerprint
from qeclang.workflow import Program
from qeclang.workflow.artifacts import describe
from qeclang.workflow.import_artifact import ImportBudget, import_artifact
from qeclang.workflow.json_input import validate_json_data
from .bundles import ArchiveLimits, retrieve
from .protocol import validate_key
from .reuse import _json_policy, _map_admission, _target_payload


def _relation_admission(relation, target, limits):
    validate_json_data(relation, limits=_json_policy(limits))
    if type(relation) is not dict:
        raise ValueError("distance reuse requires an explicit metric relation")
    if relation.get("relation") == "permutation":
        if set(relation) != {"relation", "permutation"}:
            raise ValueError("permutation relation has unsupported fields")
        permutation = relation["permutation"]
        if (type(permutation) is not list or len(permutation) != target.n
                or any(type(index) is not int for index in permutation)
                or sorted(permutation) != list(range(target.n))):
            raise ValueError("permutation must be a bijection of the physical sites")
    elif relation.get("relation") == "generator_spaces":
        if set(relation) != {"relation", "maps"}:
            raise ValueError("generator-space relation has unsupported fields")
        _map_admission(relation["maps"], limits)
    else:
        raise ValueError("distance reuse supports permutation or generator_spaces relations")


class _CheckedArchive:
    def __init__(self, received, limits):
        self.program = Program.from_dict(received["program"], limits=limits.program)
        self.nodes = {node.id: node for node in self.program.nodes}
        self.completed = {step["node"]: step for step in received["run"]["trace"]
                          if step["status"] == "completed"}
        self.artifacts = received["run"]["artifacts"]
        self.claims = received["inspection"]["claims"]
        self.policy = limits.inspection.artifacts
        self.budget = ImportBudget(max_artifact_bytes=self.policy.max_bytes,
                                  max_validation_work=self.policy.max_validation_work)
        self.cache = {}

    def node(self, identifier):
        if identifier not in self.nodes or identifier not in self.completed:
            raise ValueError("distance premise is outside the completed checked archive prefix")
        return self.nodes[identifier]

    def digest(self, identifier):
        self.node(identifier)
        return self.completed[identifier]["output"]

    def value(self, identifier, kind):
        node = self.node(identifier)
        if node.result_type != kind:
            raise ValueError("archived distance premise has an incompatible artifact kind")
        key = self.digest(identifier)
        if (key, kind) not in self.cache:
            self.cache[key, kind] = import_artifact(self.artifacts[key], kind,
                                                   limits=self.policy, budget=self.budget)
        return self.cache[key, kind]

    def claim(self, identifier, predicate):
        claims = [claim for claim in self.claims if claim.get("node") == identifier
                  and claim.get("scope") == "instance" and claim.get("predicate") == predicate]
        if len(claims) != 1 or claims[0]["evidence"] != self.digest(identifier):
            raise ValueError("selected premise has no unique independently checked matching claim")
        return claims[0]


def _interval(view, identifier, source_id, subject, sector):
    node = view.node(identifier)
    predicate = "logical_distance" if sector == "pauli" else "sector_distance"
    claim = view.claim(identifier, predicate)
    value = view.value(identifier, "distance_claim")
    if (node.code_id != source_id or node.sector != sector or value["source"] != subject
            or value["sector"] != sector
            or any(claim.get(key) != value[key]
                   for key in ("source", "sector", "lower_bound", "upper_bound", "status"))):
        raise ValueError("distance interval differs from its checked source or sector")
    if claim.get("premises") != [view.digest(item) for item in node.inputs[1:]]:
        raise ValueError("distance interval is not bound to its checked premise artifacts")
    return value


def _sector_premises(view, identifier, source_id, subject):
    node = view.node(identifier)
    if node.op != "css.combine_distance":
        raise ValueError("distance reuse requires native css.combine_distance sector premises; "
                         "transported interval ancestry is not reconstructed")
    sector = node.sector
    value = _interval(view, identifier, source_id, subject, sector)
    exclusion_id, upper_id = node.inputs[1:]
    exclusion_node, upper_node = view.node(exclusion_id), view.node(upper_id)
    if (node.inputs[0] != source_id or exclusion_node.code_id != source_id
            or upper_node.code_id != source_id or exclusion_node.sector != sector
            or upper_node.sector != sector):
        raise ValueError("distance premises must share the exact source node and sector")
    exclusion = view.value(exclusion_id, "exclusion")
    upper = view.value(upper_id, "witness")
    low_fact = view.claim(exclusion_id, "nontrivial_weight_exclusion")
    high_fact = view.claim(upper_id, "distance_upper_bound")
    if (exclusion["status"] != "excluded" or exclusion["source"] != subject
            or exclusion["sector"] != sector or upper["kind"] != "css_logical"
            or upper["source"] != subject or upper["claim"]["sector"] != sector
            or low_fact["source"] != subject or low_fact["sector"] != sector
            or high_fact["source"] != subject or high_fact["sector"] != sector
            or low_fact["through"] != exclusion["through"]
            or high_fact["bound"] != upper["claim"]["distance_upper_bound"]
            or value["lower_bound"] != exclusion["through"] + 1
            or value["upper_bound"] != upper["claim"]["distance_upper_bound"]):
        raise ValueError("archived interval lacks matching actual exclusion and upper evidence")
    return {"sector": sector, "exclusion": exclusion, "upper": upper,
            "exclusion_node": exclusion_id, "upper_node": upper_id, "value": value}


def _select(view, identifier):
    node = view.node(identifier)
    if node.op not in ("css.combine_distance", "css.full_distance"):
        raise ValueError("distance reuse supports native css.combine_distance or css.full_distance "
                         "claims; transported interval ancestry is not reconstructed")
    source = view.value(node.code_id, "css")
    subject = presentation_fingerprint(source)
    if node.op == "css.combine_distance":
        return source, [_sector_premises(view, identifier, node.code_id, subject)]
    value = _interval(view, identifier, node.code_id, subject, "pauli")
    parts = [_sector_premises(view, child, node.code_id, subject) for child in node.inputs[1:]]
    if ({part["sector"] for part in parts} != {"x", "z"}
            or value["lower_bound"] != min(part["value"]["lower_bound"] for part in parts)
            or value["upper_bound"] != min(part["value"]["upper_bound"] for part in parts)):
        raise ValueError("full distance lacks both matching checked sector premises")
    return source, sorted(parts, key=lambda part: part["sector"])


def make_distance_reuse_program(store, key, distance_node, target, relation, *,
                                allow_prefix=False, limits=ArchiveLimits()) -> Program:
    """Rebuild checked native premises, then request explicit metric transport.

    target must be CSSCode. A selected sector combine or full-distance join may
    use imported exclusion/witness ancestors, but transported interval ancestry
    is deliberately unsupported. Target relation equations are checked during
    mandatory execution/inspection, not assembly. Origin fields are hints.
    """
    if not isinstance(limits, ArchiveLimits) or type(allow_prefix) is not bool:
        raise TypeError("distance reuse requires ArchiveLimits and a boolean prefix policy")
    validate_key(key)
    if (type(distance_node) is not str or not distance_node
            or len(distance_node) > limits.program.max_id_length):
        raise ValueError("distance node must be a bounded nonempty identifier")
    target_record = _target_payload(target, limits)
    _relation_admission(relation, target, limits)
    received = retrieve(store, key, limits=limits, allow_prefix=allow_prefix)
    source, parts = _select(_CheckedArchive(received, limits), distance_node)
    if source.n != target.n:
        raise ValueError("distance transport requires identical physical coordinate width")
    source_record = describe("css", source, limits=_json_policy(limits))["payload"]
    nodes = [dict(id="source", op="css.literal", inputs=[], params={"code": source_record}),
             dict(id="target", op="css.literal", inputs=[], params={"code": target_record})]
    for part in parts:
        sector = part["sector"]
        for suffix, op, param, record, origin in (
                ("exclusion", "css.import_exclusion", "record", part["exclusion"], part["exclusion_node"]),
                ("upper", "css.import_witness", "certificate", part["upper"], part["upper_node"])):
            nodes.append(dict(id=f"{sector}_{suffix}", op=op, inputs=["source"],
                              params={param: record, "origin": {"bundle_id": key, "node": origin}}))
        nodes.append(dict(id=f"{sector}_distance", op="css.combine_distance",
                          inputs=["source", f"{sector}_exclusion", f"{sector}_upper"], params={}))
    claim_node = f"{parts[0]['sector']}_distance"
    if len(parts) == 2:
        claim_node = "source_distance"
        nodes.append(dict(id=claim_node, op="css.full_distance",
                          inputs=["source", "x_distance", "z_distance"], params={}))
    nodes.append(dict(id="transported_distance", op="css.transport_distance",
                      inputs=["source", "target", claim_node], params=relation))
    return Program.from_dict({"schema": "qeclang.program.v1", "nodes": nodes,
                              "outputs": ["target", "transported_distance"]}, limits=limits.program)


__all__ = ["make_distance_reuse_program"]
