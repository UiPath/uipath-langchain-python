"""Tests for Data Fabric models — FieldSchema enrichment."""

from uipath_langchain.agent.tools.datafabric_tool.models import FieldSchema


class TestFieldSchemaKeyMarker:
    def test_primary_key(self):
        f = FieldSchema(name="id", type="int", is_primary_key=True)
        assert f.key_marker == "PK"

    def test_foreign_key(self):
        f = FieldSchema(name="customer_id", type="int", is_foreign_key=True)
        assert f.key_marker == "FK"

    def test_primary_key_takes_precedence(self):
        f = FieldSchema(name="id", type="int", is_primary_key=True, is_foreign_key=True)
        assert f.key_marker == "PK"

    def test_no_key(self):
        f = FieldSchema(name="name", type="varchar")
        assert f.key_marker == ""


class TestFieldSchemaShortDescription:
    def test_with_description(self):
        f = FieldSchema(name="x", type="int", description="The main identifier")
        assert f.short_description == "The main identifier"

    def test_long_description_truncated(self):
        long_desc = "A" * 100
        f = FieldSchema(name="x", type="int", description=long_desc)
        assert len(f.short_description) <= 84  # 80 + "..."
        assert f.short_description.endswith("...")

    def test_fallback_to_display_name(self):
        f = FieldSchema(name="cust_id", type="int", display_name="Customer ID")
        assert f.short_description == "Customer ID"

    def test_no_fallback_when_display_name_matches(self):
        f = FieldSchema(name="name", type="varchar", display_name="name")
        assert f.short_description == ""

    def test_empty_when_nothing_available(self):
        f = FieldSchema(name="x", type="int")
        assert f.short_description == ""
