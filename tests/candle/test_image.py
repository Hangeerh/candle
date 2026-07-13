from pathlib import Path

import pytest

from candle.image import Image, ImageCollection, _is_valid_extension, _norm_path


def test_norm_path_from_str():
    assert _norm_path("foo/bar") == Path("foo/bar")


def test_norm_path_from_path():
    p = Path("foo/bar")
    assert _norm_path(p) is p


def test_is_valid_extension():
    assert _is_valid_extension("photo.jpg") is True
    assert _is_valid_extension("photo.png") is True
    assert _is_valid_extension("photo.jpeg") is True
    assert _is_valid_extension("photo.txt") is False
    assert _is_valid_extension("photo") is False
    assert _is_valid_extension("photo.gif") is False


def test_image_properties():
    path = Path("some/dir/photo.jpg")
    img = Image(path)
    assert img.file == "photo.jpg"
    assert img.dir == Path("some/dir")
    assert img.new_name is None


def test_image_collection_adds_valid_images():
    col = ImageCollection()
    col.add("photo.jpg")
    col.add("photo.png")
    col.add("photo.jpeg")
    assert len(col.images) == 3
    assert all(isinstance(img, Image) for img in col.images)


def test_image_collection_skips_invalid_extensions():
    col = ImageCollection()
    col.add("photo.txt")
    col.add("photo")
    col.add("photo.gif")
    assert col.images == []


def test_image_collection_add_dir_requires_directory():
    col = ImageCollection()
    with pytest.raises(RuntimeError, match="Provided path is not a directory"):
        col.add_dir("nonexistent_path")


def test_image_collection_add_dir_flat(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jpg").write_text("a")
    (src / "b.png").write_text("b")
    (src / "c.txt").write_text("c")

    col = ImageCollection()
    col.add_dir(src)
    assert len(col.images) == 2
    assert {img.file for img in col.images} == {"a.jpg", "b.png"}


def test_image_collection_add_dir_recursive(tmp_path: Path):
    src = tmp_path / "src"
    sub = src / "sub"
    sub.mkdir(parents=True)
    (src / "a.jpg").write_text("a")
    (sub / "b.png").write_text("b")

    col = ImageCollection()
    col.add_dir(src, recursive=True)
    assert len(col.images) == 2
    assert {img.file for img in col.images} == {"a.jpg", "b.png"}


def test_image_collection_add_dir_non_recursive(tmp_path: Path):
    src = tmp_path / "src"
    sub = src / "sub"
    sub.mkdir(parents=True)
    (src / "a.jpg").write_text("a")
    (sub / "b.png").write_text("b")

    col = ImageCollection()
    col.add_dir(src, recursive=False)
    assert len(col.images) == 1
    assert col.images[0].file == "a.jpg"


def test_image_collection_save(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jpg").write_text("a")

    dst = tmp_path / "dst"

    col = ImageCollection()
    col.add(src / "a.jpg")
    col.save(dst)

    assert (dst / "a.jpg").exists()
    assert (dst / "a.jpg").read_text() == "a"
