"""Bind lifted-parameter artifacts to a node and replay explicitly requested proofs."""
from dataclasses import dataclass

from qeclang.constructions.records import fingerprint
from .lift_relations import comparison_units, check_lift


class InspectionQuota(Exception):
    pass


@dataclass
class LiftInspection:
    remaining_steps: int
    remaining_comparison_units: int

    def expansion(self, lift, target):
        units = comparison_units(lift)
        if units > self.remaining_comparison_units:
            raise InspectionQuota('lift expansion relation exceeds reader work budget')
        self.remaining_comparison_units -= units
        return check_lift(lift, target)

    def parameters(self, node, lift, result, key, limits):
        from .rank_proof import verify_dimension_proof
        source = fingerprint(lift)
        params = node.params
        if (result['n'] != lift.n or result['method'] != params['backend']
                or result['construction_fingerprint'] != source):
            raise ValueError('parameter record differs from construction, method or blocklength')
        if result['steps'] is not None and result['steps'] > params['max_steps']:
            raise ValueError('parameter report exceeds requested arithmetic budget')
        if result['status'] != 'exact' or not params['record_operations']:
            return None
        proof = verify_dimension_proof(lift, result, max_steps=self.remaining_steps,
                                       max_blocks=limits.max_blocks,
                                       max_degree=limits.max_lift_degree,
                                       max_polynomial_degree=limits.max_polynomial_degree,
                                       max_operations=limits.max_polynomial_operations,
                                       max_entries=limits.max_blocks, max_symbolic_terms=limits.max_symbolic_terms)
        self.remaining_steps -= proof.steps
        if proof.status == 'budget_exhausted':
            raise InspectionQuota(proof.reason)
        if proof.status != 'verified':
            raise ValueError(proof.reason or 'invalid dimension proof')
        return {'node': node.id, 'scope': 'instance', 'source': source, 'predicate': 'logical_dimension',
                'value': proof.k, 'rank_x': proof.rank_x, 'rank_z': proof.rank_z, 'evidence': key,
                'verification_steps': proof.steps}
