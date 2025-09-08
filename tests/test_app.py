import pytest

from src.app import read_urls


def test_read_urls_valid_file(tmp_path):
    # Create a temporary file with URLs
    file = tmp_path / "urls.txt"
    file.write_text("http://example.com\nhttps://test.com\n")
    result = read_urls(str(file))
    assert result == ["http://example.com", "https://test.com"]


def test_read_urls_empty_lines(tmp_path):
    file = tmp_path / "urls.txt"
    file.write_text("\nhttp://example.com\n\n")
    result = read_urls(str(file))
    assert result == ["http://example.com"]


def test_read_urls_file_not_found():
    with pytest.raises(SystemExit) as e:
        read_urls("nonexistent.txt")
    assert e.type == SystemExit
    assert e.value.code == 1
