import json
from pathlib import Path

import pytest

from candle.data.image import ImageCollection, _is_valid_img_extension, _norm_path


def test_norm_path_from_str():
    assert _norm_path("foo/bar") == Path("foo/bar")


def test_norm_path_from_path():
    path = Path("foo/bar")
    assert _norm_path(path) is path


def test_is_valid_img_extension():
    assert _is_valid_img_extension("photo.jpg") is True
    assert _is_valid_img_extension("photo.png") is True
    assert _is_valid_img_extension("photo.jpeg") is True
    assert _is_valid_img_extension("photo.txt") is False
    assert _is_valid_img_extension("photo") is False
    assert _is_valid_img_extension("photo.gif") is False


def test_image_collection_adds_valid_images_and_attributes():
    collection = ImageCollection()
    collection.add("photo.jpg", {"label": "cat"})
    collection.add("photo.png")
    collection.add("photo.jpeg")
    collection.add("photo.gif")

    assert len(collection) == 3
    assert [
        image.source.file_name + image.source.file_extens for image in collection
    ] == [
        "photo.jpg",
        "photo.png",
        "photo.jpeg",
    ]
    assert collection[0].attribute == {"label": "cat"}
    assert collection[1].attribute is None


def test_image_collection_getitem_supports_list_indexing():
    collection = ImageCollection()
    collection.add("first.jpg")
    collection.add("second.jpg")

    assert collection[0] is collection.images[0]
    assert collection[-1] is collection.images[-1]
    assert collection[:1] == collection.images[:1]


def test_image_collection_add_dir_requires_directory(tmp_path: Path):
    collection = ImageCollection()
    with pytest.raises(RuntimeError, match="Provided path is not a directory"):
        collection.add_dir(tmp_path / "missing")


def test_image_collection_add_dir_loads_images_and_json_attributes(tmp_path: Path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "a.jpg").write_text("a")
    (source_dir / "a.json").write_text(json.dumps({"label": "cat"}))
    (source_dir / "b.png").write_text("b")
    (source_dir / "c.txt").write_text("c")
    (source_dir / "orphan.json").write_text(json.dumps({"label": "orphan"}))

    collection = ImageCollection()
    collection.add_dir(source_dir)

    assert len(collection) == 2
    assert {image.source.file_name for image in collection} == {"a", "b"}
    assert next(
        image for image in collection if image.source.file_name == "a"
    ).attribute == {"label": "cat"}
    assert (
        next(image for image in collection if image.source.file_name == "b").attribute
        is None
    )


def test_image_collection_add_dir_recurses_when_requested(tmp_path: Path):
    source_dir = tmp_path / "src"
    sub_dir = source_dir / "sub"
    sub_dir.mkdir(parents=True)
    (source_dir / "a.jpg").write_text("a")
    (sub_dir / "b.png").write_text("b")

    collection = ImageCollection()
    collection.add_dir(source_dir, recursive=True)

    assert {image.source.file_name for image in collection} == {"a", "b"}


def test_image_collection_add_dir_does_not_recurse_by_default(tmp_path: Path):
    source_dir = tmp_path / "src"
    sub_dir = source_dir / "sub"
    sub_dir.mkdir(parents=True)
    (source_dir / "a.jpg").write_text("a")
    (sub_dir / "b.png").write_text("b")

    collection = ImageCollection()
    collection.add_dir(source_dir)

    assert len(collection) == 1
    assert collection[0].source.file_name == "a"


def test_image_collection_add_attributes_and_override():
    collection = ImageCollection()
    collection.add("a.jpg", {"label": "original"})
    collection.add("b.png")

    collection.add_attributes(
        [("a.jpg", {"label": "updated"}), ("b.png", {"label": "new"})]
    )
    assert collection[0].attribute == {"label": "original"}
    assert collection[1].attribute == {"label": "new"}

    collection.add_attributes(
        [("a.jpg", {"label": "updated"}), ("missing.jpg", {"label": "ignored"})],
        override_existing_attributes=True,
    )
    assert collection[0].attribute == {"label": "updated"}
    assert len(collection) == 2


def test_image_collection_filter_sets_keep_flags():
    collection = ImageCollection()
    collection.add("keep.jpg")
    collection.add("discard.jpg")

    collection.filter(lambda image: image.source.file_name == "keep")

    assert [image.keep for image in collection] == [True, False]


def test_image_collection_merge_accepts_collection_or_list():
    first = ImageCollection()
    first.add("first.jpg")
    second = ImageCollection()
    second.add("second.jpg")
    third = ImageCollection()
    third.add("third.jpg")

    first.merge(second)
    first.merge([third])

    assert len(first) == 3
    assert first[1] is second[0]
    assert first[2] is third[0]


def test_image_collection_batch_rename_numbers_kept_images():
    collection = ImageCollection()
    collection.add("first.jpg")
    collection.add("second.jpg")
    collection.add("third.jpg")
    collection.filter(lambda image: image.source.file_name != "second")

    collection.batch_rename()

    assert [image.new_name for image in collection] == ["0", None, "2"]


def test_image_collection_save_copies_only_kept_images(tmp_path: Path, monkeypatch):
    collection = ImageCollection()
    collection.add("first.jpg")
    collection.add("second.jpg")
    collection.add("third.jpg")
    collection.filter(lambda image: image.source.file_name != "second")
    copied = []

    for image in collection:
        monkeypatch.setattr(
            image,
            "copy_to_dir",
            lambda directory, image=image: copied.append(
                (image.source.file_name, directory)
            ),
        )

    destination = tmp_path / "dst"
    collection.save(destination, images_per_dir=None)

    assert destination.is_dir()
    assert copied == [("first", destination), ("third", destination)]


def test_image_collection_save_shards_images_into_subdirectories(
    tmp_path: Path, monkeypatch
):
    collection = ImageCollection()
    collection.add("first.jpg")
    collection.add("second.jpg")
    collection.add("third.jpg")
    copied = []

    for image in collection:
        monkeypatch.setattr(
            image,
            "copy_to_dir",
            lambda directory, image=image: copied.append(
                (image.source.file_name, directory)
            ),
        )

    destination = tmp_path / "dst"
    collection.save(destination, images_per_dir=2)

    assert copied == [
        ("first", destination / "00000"),
        ("second", destination / "00000"),
        ("third", destination / "00001"),
    ]
    assert all(directory.is_dir() for _, directory in copied)


def test_image_collection_save_requires_positive_shard_size(tmp_path: Path):
    collection = ImageCollection()
    collection.add("photo.jpg")

    with pytest.raises(AssertionError, match="images_per_dir must be greater than 0"):
        collection.save(tmp_path / "dst", images_per_dir=0)
