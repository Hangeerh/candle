from candle.utility.file import (
    _is_macos_metadata,
    _is_valid_img_extension,
    _norm_path,
)

from pathlib import Path


def test_utility_helpers():
    path = Path("foo/bar")
    assert _norm_path(path) is path
    assert _norm_path("foo/bar") == Path("foo/bar")

    assert _is_valid_img_extension("photo.jpg") is True
    assert _is_valid_img_extension("photo.jpeg") is True
    assert _is_valid_img_extension("photo.png") is True
    assert _is_valid_img_extension("photo.gif") is False
    assert _is_valid_img_extension("photo") is False
    assert _is_valid_img_extension("photo.txt") is False

    assert _is_macos_metadata("._photo.jpg") is True
    assert _is_macos_metadata("nested/._photo.jpg") is True
    assert _is_macos_metadata("__MACOSX/photo.jpg") is True
    assert _is_macos_metadata("nested/__MACOSX/photo.jpg") is True
    assert _is_macos_metadata("photo.jpg") is False
