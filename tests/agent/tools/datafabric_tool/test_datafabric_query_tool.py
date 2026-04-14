"""Tests for Data Fabric SQL validation."""

from uipath_langchain.agent.tools.datafabric_query_tool import _validate_sql


class TestValidateSql:
    def test_valid_simple_select(self):
        assert _validate_sql("SELECT id, name FROM Customer WHERE id = 1") is None

    def test_valid_with_limit(self):
        assert _validate_sql("SELECT id FROM Customer LIMIT 100") is None

    def test_valid_with_where(self):
        assert _validate_sql("SELECT id FROM Customer WHERE name = 'foo'") is None

    def test_valid_aggregation(self):
        sql = "SELECT country, COUNT(id) FROM Customer GROUP BY country LIMIT 10"
        assert _validate_sql(sql) is None

    def test_valid_left_join(self):
        sql = "SELECT o.id, c.name FROM Orders o LEFT JOIN Customer c ON o.cid = c.id WHERE o.id = 1"
        assert _validate_sql(sql) is None

    def test_empty_sql(self):
        error = _validate_sql("")
        assert error is not None
        assert "Empty" in error

    def test_reject_select_star(self):
        error = _validate_sql("SELECT * FROM Customer")
        assert error is not None
        assert "SELECT *" in error

    def test_reject_count_star(self):
        error = _validate_sql("SELECT COUNT(*) FROM Customer LIMIT 1")
        assert error is not None
        assert "COUNT(*)" in error

    def test_reject_count_one(self):
        error = _validate_sql("SELECT COUNT(1) FROM Customer LIMIT 1")
        assert error is not None
        assert "COUNT(1)" in error

    def test_reject_right_join(self):
        error = _validate_sql(
            "SELECT a.id FROM A a RIGHT JOIN B b ON a.id = b.id WHERE a.id = 1"
        )
        assert error is not None
        assert "RIGHT JOIN" in error

    def test_reject_full_outer_join(self):
        error = _validate_sql(
            "SELECT a.id FROM A a FULL OUTER JOIN B b ON a.id = b.id WHERE a.id = 1"
        )
        assert error is not None
        assert "FULL OUTER JOIN" in error

    def test_reject_cross_join(self):
        error = _validate_sql("SELECT a.id FROM A a CROSS JOIN B b WHERE a.id = 1")
        assert error is not None
        assert "CROSS JOIN" in error

    def test_reject_insert(self):
        error = _validate_sql("INSERT INTO Customer VALUES (1, 'foo')")
        assert error is not None
        assert "SELECT" in error

    def test_reject_update(self):
        error = _validate_sql("UPDATE Customer SET name = 'foo' WHERE id = 1")
        assert error is not None
        assert "SELECT" in error

    def test_reject_delete(self):
        error = _validate_sql("DELETE FROM Customer WHERE id = 1")
        assert error is not None
        assert "SELECT" in error

    def test_reject_drop(self):
        error = _validate_sql("DROP TABLE Customer")
        assert error is not None
        assert "SELECT" in error

    def test_reject_no_where_no_limit(self):
        error = _validate_sql("SELECT id, name FROM Customer")
        assert error is not None
        assert "LIMIT" in error

    def test_select_star_case_insensitive(self):
        error = _validate_sql("select * from Customer")
        assert error is not None
        assert "SELECT *" in error
