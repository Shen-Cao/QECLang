"""Bounded, strict, data-only JSON admission shared by workflow records.

Byte/count limits exclude caller-owned input and Python parser/object overhead.
Text preflight bounds nesting and token count before JSON tree construction.
"""
from dataclasses import dataclass, fields
import json
from math import isfinite
from pathlib import Path


class JSONLimitExceeded(ValueError):
    """A well-defined JSON admission budget was exceeded."""


def integer(value, name, *, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be a nonnegative integer within its limit")
    return value


@dataclass(frozen=True)
class JSONLimits:
    max_bytes: int = 10_000_000
    max_payload_items: int = 1_000_000
    max_payload_depth: int = 64
    max_string_length: int = 8_192
    max_integer_bits: int = 64

    def __post_init__(self):
        for item in fields(self):
            integer(getattr(self, item.name), item.name)
        if self.max_payload_depth > 128:
            raise ValueError("max_payload_depth must not exceed 128")
        if self.max_integer_bits > 4_096:
            raise ValueError("max_integer_bits must not exceed 4096")


def canonical_json(value):
    """Serialize admitted JSON data; callers choose admission limits separately."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _check_limits(limits):
    if not isinstance(limits, JSONLimits):
        raise TypeError("limits must be JSONLimits")


def _payload(value, limits):
    """Check direct Python input before whole-JSON payload copying/serialization."""
    items = size = 0
    active = set()
    def charge(count=0, bytes_count=0):
        nonlocal items, size
        items += count
        size += bytes_count
        if items > limits.max_payload_items:
            raise JSONLimitExceeded("JSON payload exceeds max_payload_items")
        if size > limits.max_bytes:
            raise JSONLimitExceeded("canonical JSON payload exceeds max_bytes")
    def visit(value, depth):
        charge(1)
        if type(value) in (dict, list):
            if depth + 1 > limits.max_payload_depth:
                raise JSONLimitExceeded("JSON payload exceeds max_payload_depth")
            if id(value) in active:
                raise ValueError("cyclic Python containers are not JSON data")
            needed = len(value) * (2 if type(value) is dict else 1)
            if needed > limits.max_payload_items - items:
                raise JSONLimitExceeded("JSON payload exceeds max_payload_items")
            active.add(id(value))
            charge(bytes_count=2 + max(0, len(value)-1))
            if type(value) is dict:
                for key, child in value.items():
                    if type(key) is not str:
                        raise ValueError("JSON object keys must be strings")
                    visit(key, depth + 1)
                    charge(bytes_count=1)
                    visit(child, depth + 1)
            else:
                for child in value:
                    visit(child, depth + 1)
            active.remove(id(value))
        elif type(value) is str:
            if len(value) > limits.max_string_length:
                raise JSONLimitExceeded("JSON payload exceeds max_string_length")
            if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
                raise ValueError("JSON strings must contain valid Unicode scalar values")
            charge(bytes_count=len(canonical_json(value)))
        elif type(value) is int:
            if value.bit_length() > limits.max_integer_bits:
                raise JSONLimitExceeded("JSON payload integer exceeds max_integer_bits")
            charge(bytes_count=len(str(value)))
        elif value is None or type(value) is bool:
            charge(bytes_count=len(canonical_json(value)))
        elif type(value) is float and isfinite(value):
            charge(bytes_count=len(canonical_json(value)))
        else:
            raise ValueError("JSON payload must contain finite JSON data only")
    visit(value, 0)
    return size


def _text_preflight(text, limits):
    """Bound lexical nesting/token volume before the JSON parser builds a tree."""
    depth = items = 0
    quoted = escaped = token = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted, token = True, False
            items += 1
        elif char in "[{":
            depth += 1
            items += 1
            token = False
            if depth > limits.max_payload_depth:
                raise JSONLimitExceeded("JSON payload exceeds max_payload_depth")
        elif char in "]}":
            depth -= 1
            token = False
            if depth < 0:
                raise ValueError("invalid JSON container nesting")
        elif char.isspace() or char in ",:":
            token = False
        elif not token:
            items += 1
            token = True
        if items > limits.max_payload_items:
            raise JSONLimitExceeded("JSON payload exceeds max_payload_items")


def measure_json_data(data, *, limits=JSONLimits()):
    """Validate caller-owned JSON and return its canonical ASCII byte count."""
    _check_limits(limits)
    return _payload(data, limits)


def validate_json_data(data, *, limits=JSONLimits()):
    """Validate existing Python JSON containers without constructing runtime objects."""
    measure_json_data(data, limits=limits)
    return data


def loads_json(content, *, limits=JSONLimits()):
    """Parse bounded UTF-8 text, rejecting duplicate keys and nonfinite numbers."""
    _check_limits(limits)
    if type(content) is bytes:
        if len(content) > limits.max_bytes:
            raise JSONLimitExceeded("JSON input exceeds max_bytes")
        content = content.decode("utf-8")
    elif type(content) is not str:
        raise TypeError("program JSON must be str or UTF-8 bytes")
    if len(content) > limits.max_bytes or len(content.encode("utf-8")) > limits.max_bytes:
        raise JSONLimitExceeded("JSON input exceeds max_bytes")
    _text_preflight(content, limits)
    def unique(pairs):
        value = {}
        for key, child in pairs:
            if key in value:
                raise ValueError("duplicate JSON object key")
            value[key] = child
        return value
    def integer(text):
        if len(text.lstrip("-")) > (limits.max_integer_bits * 30103)//100000 + 1:
            raise JSONLimitExceeded("JSON integer exceeds max_integer_bits")
        value = int(text)
        if value.bit_length() > limits.max_integer_bits:
            raise JSONLimitExceeded("JSON integer exceeds max_integer_bits")
        return value
    def real(text):
        value = float(text)
        if not isfinite(value):
            raise ValueError("nonfinite JSON number")
        return value
    def constant(text):
        raise ValueError(f"nonfinite JSON constant: {text}")
    data = json.loads(content, object_pairs_hook=unique, parse_int=integer,
                      parse_float=real, parse_constant=constant)
    return validate_json_data(data, limits=limits)


def load_json(path, *, limits=JSONLimits()):
    """Read at most max_bytes+1 bytes, then apply strict JSON admission."""
    _check_limits(limits)
    with Path(path).open("rb") as stream:
        content = stream.read(limits.max_bytes + 1)
    if len(content) > limits.max_bytes:
        raise JSONLimitExceeded("JSON input exceeds max_bytes")
    return loads_json(content, limits=limits)


__all__ = ["JSONLimitExceeded", "JSONLimits", "loads_json", "load_json", "validate_json_data",
           "measure_json_data", "canonical_json"]
