"""Tests for utils.csv_writer — price/barcode/date/discount parsing, color classification, row creation.

These are pure functions with no external dependencies, ideal for unit testing.
"""

import pytest
from utils.csv_writer import (
    extract_price_value,
    classify_color,
    extract_barcode,
    extract_discount,
    extract_date,
    make_row,
    extract_special_symbols,
    is_barcode_like,
)


# ── extract_price_value ──────────────────────────────────────────


class TestExtractPriceValue:
    def test_single_integer_price(self):
        assert extract_price_value(["299"]) == [299.0]

    def test_single_decimal_price_dot(self):
        assert extract_price_value(["99.99"]) == [99.99]

    def test_single_decimal_price_comma(self):
        assert extract_price_value(["99,99"]) == [99.99]

    def test_multiple_prices_sorted_descending(self):
        result = extract_price_value(["150.00", "299.90", "199.50"])
        assert result == [299.90, 199.50, 150.00]

    def test_russian_format_with_space(self):
        result = extract_price_value(["1 299"])
        # regex strips non-digits, gets "1299" → 1299.0
        assert 1299.0 in result

    def test_price_with_currency_symbol(self):
        result = extract_price_value(["150р", "299 руб"])
        assert 150.0 in result
        assert 299.0 in result

    def test_barcode_like_ignored(self):
        # 12+ consecutive digits should be ignored (treated as barcode)
        assert extract_price_value(["123456789012"]) == []

    def test_below_minimum_price_ignored(self):
        assert extract_price_value(["5"]) == []

    def test_above_maximum_price_ignored(self):
        # 9999999 is 7 digits → regex matches 99999 (< 1M limit), but
        # the real code strips non-digits and matches a 5-digit group.
        # The function actually returns a valid price from the digits.
        # What matters: values > 1_000_000 are filtered.
        result = extract_price_value(["9999999"])
        assert all(v <= 1_000_000 for v in result)

    def test_empty_list(self):
        assert extract_price_value([]) == []

    def test_non_string_input(self):
        # int 123 → str(123) = "123" → parsed as price 123.0 (valid, >=10)
        # None → str(None) = "None" → no digits, ignored
        result = extract_price_value([None, 123, "99.99"])
        assert 123.0 in result
        assert 99.99 in result

    def test_mixed_text_and_prices(self):
        result = extract_price_value(["Цена: 299.50 руб", "Скидка: 10%"])
        assert 299.50 in result

    def test_duplicate_prices_deduplicated(self):
        result = extract_price_value(["99.99", "99.99"])
        assert result == [99.99]

    def test_zero_price_ignored(self):
        assert extract_price_value(["0"]) == []

    def test_negative_price_ignored(self):
        # extract_price_value strips non-digits, so "-5" → "5" → 5.0 (valid)
        # but negative values from regex: (-?\d+) would need sign handling
        # Current implementation strips "-" via re.sub(r"[^\d.,]", "")
        result = extract_price_value(["-5"])
        # After stripping: "5" → matches as integer 5, but 5 < 10 so filtered out
        assert result == []


# ── classify_color ──────────────────────────────────────────────


class TestClassifyColor:
    def test_red(self):
        assert classify_color((50, 80, 220)) == "red"

    def test_yellow(self):
        assert classify_color((40, 180, 230)) == "yellow"

    def test_orange(self):
        assert classify_color((30, 150, 220)) == "orange"

    def test_white(self):
        assert classify_color((200, 200, 210)) == "white"

    def test_light_gray_treated_as_white(self):
        assert classify_color((180, 180, 190)) == "white"

    def test_dark_color_treated_as_white(self):
        # Low R,G → falls through to default "white"
        assert classify_color((50, 50, 50)) == "white"

    def test_pure_blue_treated_as_white(self):
        # High B but low R,G → default
        assert classify_color((220, 150, 30)) == "white"


# ── extract_barcode ──────────────────────────────────────────────


class TestExtractBarcode:
    def test_ean13(self):
        # 4600644110002 is a valid EAN13 (check digit 2)
        assert extract_barcode(["4600644110002"]) == "4600644110002"

    def test_ean13_in_text(self):
        assert extract_barcode(["штрихкод 4600644110002 конец"]) == "4600644110002"

    def test_short_digits_ignored(self):
        assert extract_barcode(["12345"]) is None

    def test_no_digits(self):
        assert extract_barcode(["abc"]) is None

    def test_empty_list(self):
        assert extract_barcode([]) is None

    def test_non_string_input(self):
        assert extract_barcode([None]) is None

    def test_invalid_ean13_checksum_ignored(self):
        # 4600644110006 has invalid check digit (correct is ...0002) → ignored
        assert extract_barcode(["4600644110006"]) is None

    def test_11_digits_ignored(self):
        assert extract_barcode(["12345678901"]) is None


# ── extract_discount ────────────────────────────────────────────


class TestExtractDiscount:
    def test_positive_percent(self):
        assert extract_discount(["скидка 25%"]) == "25%"

    def test_negative_percent(self):
        assert extract_discount(["-15%"]) == "-15%"

    def test_no_percent(self):
        assert extract_discount(["скидка 25"]) is None

    def test_empty_list(self):
        assert extract_discount([]) is None

    def test_multiple_matches_returns_first(self):
        # Both have %, should return first
        result = extract_discount(["10%", "20%"])
        assert result == "10%"


# ── extract_date ────────────────────────────────────────────────


class TestExtractDate:
    def test_dd_mm_yyyy_dots(self):
        assert extract_date(["01.01.2025"]) == "01.01.2025"

    def test_dd_mm_yy_slashes(self):
        assert extract_date(["15/06/26"]) == "15/06/26"

    def test_no_date(self):
        assert extract_date(["текст без даты"]) is None

    def test_empty_list(self):
        assert extract_date([]) is None

    def test_date_in_longer_text(self):
        assert extract_date(["действует до 31.12.2025 г."]) == "31.12.2025"


# ── make_row ────────────────────────────────────────────────────


class TestMakeRow:
    def test_all_fields_empty_by_default(self):
        row = make_row()
        for col in [
            "filename", "product_name", "price_default", "price_card",
            "price_discount", "barcode", "discount_amount", "id_sku",
            "print_datetime", "code", "additional_info", "color",
            "special_symbols", "frame_timestamp",
            "x_min", "y_min", "x_max", "y_max",
            "qr_code_barcode",
            "price1_qr", "price2_qr", "price3_qr", "price4_qr",
            "wholesale_level_1_count", "wholesale_level_1_price",
            "wholesale_level_2_count", "wholesale_level_2_price",
            "action_price_qr", "action_code_qr", "raw_qr_data",
        ]:
            assert row.get(col) == "", f"Field '{col}' should be empty string"

    def test_kwargs_override_defaults(self):
        row = make_row(product_name="Молоко", price_default=99.90)
        assert row["product_name"] == "Молоко"
        assert row["price_default"] == 99.90

    def test_unknown_kwargs_accepted(self):
        # make_row uses row.update(kwargs) — unknown keys are accepted
        row = make_row(unknown_field="value")
        assert row["unknown_field"] == "value"


# ── extract_special_symbols ────────────────────────────────────


class TestExtractSpecialSymbols:
    def test_red_color(self):
        result = extract_special_symbols(["текст"], (50, 80, 220))
        assert "red" in result

    def test_yellow_color(self):
        result = extract_special_symbols(["текст"], (40, 180, 230))
        assert "yellow" in result

    def test_sh_symbol(self):
        result = extract_special_symbols(["Шт"], (200, 200, 210))
        assert "Ш" in result

    def test_percent_symbol(self):
        result = extract_special_symbols(["скидка 25%"], (200, 200, 210))
        assert "%" in result

    def test_no_symbols(self):
        result = extract_special_symbols(["обычный текст"], (200, 200, 210))
        assert result is None


# ── is_barcode_like ────────────────────────────────────────────


class TestIsBarcodeLike:
    def test_12_digits(self):
        assert is_barcode_like("4600644110006") is True

    def test_short_string(self):
        assert is_barcode_like("123") is False

    def test_non_string(self):
        assert is_barcode_like(None) is False

    def test_mixed(self):
        assert is_barcode_like("abc123456789012") is True

    def test_11_digits(self):
        assert is_barcode_like("12345678901") is False
