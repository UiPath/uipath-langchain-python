import re

from uipath_agents._bts.bts_helpers import (
    extract_transaction_id,
    generate_guid,
    generate_operation_id,
)

GUID_HEX_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def test_generate_guid_returns_32_hex_chars() -> None:
    guid = generate_guid()
    assert GUID_HEX_PATTERN.match(guid)


def test_generate_guid_no_hyphens() -> None:
    guid = generate_guid()
    assert "-" not in guid


def test_generate_guid_unique() -> None:
    assert generate_guid() != generate_guid()


def test_generate_operation_id_format() -> None:
    txn_id = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"
    op_id = generate_operation_id(txn_id)
    parts = op_id.split("-")
    assert len(parts) == 2
    assert parts[0] == txn_id
    assert GUID_HEX_PATTERN.match(parts[1])


def test_extract_transaction_id_from_operation_id() -> None:
    txn_id = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"
    op_id = f"{txn_id}-deadbeef01234567890abcdef1234567"
    assert extract_transaction_id(op_id) == txn_id


def test_roundtrip_generate_then_extract() -> None:
    txn_id = generate_guid()
    op_id = generate_operation_id(txn_id)
    assert extract_transaction_id(op_id) == txn_id
