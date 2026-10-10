from pathlib import Path
from app import format_utc
from app import parse_report


def test_app_file_exists():
    assert Path("app.py").is_file()


def test_format_utc():
    result = format_utc("2026-10-10T23:30:46Z")
    assert result == "Oct 10, 2026 · 23:30:46 UTC"

def test_invalid_timestamp():
    assert format_utc("not-a-date") == "not-a-date"

def test_parse_report():
    result = parse_report("satellite_a: ISS\nwindow_hours: 24")
    assert result["satellite_a"] == "ISS"
    assert result["window_hours"] == "24"

def test_parse_empty_report():
    assert parse_report("") == {}

def test_parse_report_ignores_invalid_lines():
    result = parse_report("ISS\nsatellite_a: ISS")
    assert result == {"satellite_a": "ISS"}