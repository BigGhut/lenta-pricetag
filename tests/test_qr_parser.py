"""Tests for utils.qr_parser — QR content parsing (JSON, key=value, pipe, raw).
"""

import pytest
from utils.qr_parser import parse_qr_content


# ── parse_qr_content ───────────────────────────────────────────


class TestParseQrContent:
    def test_json_object(self):
        raw = '{"price": "299.90", "sku": "123456789"}'
        result = parse_qr_content(raw)
        assert result["price"] == "299.90"
        assert result["sku"] == "123456789"

    def test_json_with_nested(self):
        raw = '{"data": {"name": "test"}}'
        result = parse_qr_content(raw)
        assert isinstance(result, dict)
        assert result["data"]["name"] == "test"

    def test_key_value_semicolon(self):
        raw = "price=299.90;sku=123456789;discount=15%"
        result = parse_qr_content(raw)
        assert result["price"] == "299.90"
        assert result["sku"] == "123456789"
        assert result["discount"] == "15%"

    def test_key_value_newline(self):
        raw = "price=150.00\nsku=987654321"
        result = parse_qr_content(raw)
        assert result["price"] == "150.00"
        assert result["sku"] == "987654321"

    def test_pipe_separated(self):
        raw = "field1|field2|field3"
        result = parse_qr_content(raw)
        assert result["field_0"] == "field1"
        assert result["field_1"] == "field2"
        assert result["field_2"] == "field3"

    def test_raw_string(self):
        raw = "some random string without separators"
        result = parse_qr_content(raw)
        assert result["raw"] == "some random string without separators"

    def test_empty_string(self):
        assert parse_qr_content("") == {}

    def test_none_input(self):
        assert parse_qr_content(None) == {}

    def test_invalid_json_falls_through(self):
        raw = '{"broken json'
        result = parse_qr_content(raw)
        # Not valid JSON, not key=value (no "="), not pipe-separated
        assert result["raw"] == '{"broken json'
