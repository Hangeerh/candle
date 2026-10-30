import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image as PILImage

from candle.data.image import (
    ImageCollection,
    ImageSourceTar,
    _is_valid_img_extension,
    _norm_path,
)


def _add_tar_member(tar: tarfile.TarFile, name: str, data: bytes):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", (2, 3), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


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


def test_image_collection_add_ignores_macos_metadata():
    collection = ImageCollection()
    collection.add("._photo.jpg")
    collection.add("photo.jpg")

    assert [image.source.file_name for image in collection] == ["photo"]


def test_image_collection_add_dir_ignores_macos_metadata(tmp_path: Path):
    source_dir = tmp_path / "src"
    metadata_dir = source_dir / "__MACOSX"
    metadata_dir.mkdir(parents=True)
    (source_dir / "a.png").write_text("a")
    (source_dir / "a.json").write_text(json.dumps({"label": "cat"}))
    (source_dir / "._a.png").write_bytes(b"ghost")
    (source_dir / "._a.json").write_bytes(b"not json")
    (metadata_dir / "._b.jpg").write_bytes(b"ghost")

    collection = ImageCollection()
    collection.add_dir(source_dir, recursive=True)

    assert len(collection) == 1
    assert collection[0].source.file_name == "a"
    assert collection[0].attribute == {"label": "cat"}


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


def test_image_collection_add_tar_requires_file(tmp_path: Path):
    collection = ImageCollection()
    with pytest.raises(RuntimeError, match="Provided path is not a file"):
        collection.add_tar(tmp_path / "missing.tar")


def test_image_collection_add_tar_loads_images_and_json_attributes(tmp_path: Path):
    tar_path = tmp_path / "images.tar"
    with tarfile.open(tar_path, "w") as tar:
        _add_tar_member(tar, "a.png", b"a")
        _add_tar_member(tar, "a.json", json.dumps({"label": "cat"}).encode())
        _add_tar_member(tar, "b.jpg", b"b")
        _add_tar_member(tar, "c.txt", b"c")
        _add_tar_member(tar, "orphan.json", json.dumps({"label": "orphan"}).encode())

    collection = ImageCollection()
    collection.add_tar(tar_path)

    assert len(collection) == 2
    assert {image.source.file_name for image in collection} == {"a", "b"}
    assert all(isinstance(image.source, ImageSourceTar) for image in collection)
    assert {image.source.path_in_tar for image in collection} == {"a.png", "b.jpg"}
    assert next(
        image for image in collection if image.source.file_name == "a"
    ).attribute == {"label": "cat"}
    assert (
        next(image for image in collection if image.source.file_name == "b").attribute
        is None
    )


def test_image_collection_add_tar_ignores_macos_metadata(tmp_path: Path):
    tar_path = tmp_path / "images.tar"
    with tarfile.open(tar_path, "w") as tar:
        _add_tar_member(tar, "00000.jpg", b"image")
        _add_tar_member(tar, "00000.json", json.dumps({"label": "cat"}).encode())
        _add_tar_member(tar, "._00000.jpg", b"ghost")
        _add_tar_member(tar, "._00000.json", b"not json")
        _add_tar_member(tar, "__MACOSX/._00000.jpg", b"ghost")
        _add_tar_member(tar, "__MACOSX/00000/._00000.jpg", b"ghost")

    collection = ImageCollection()
    collection.add_tar(tar_path)

    assert len(collection) == 1
    assert collection[0].source.path_in_tar == "00000.jpg"
    assert collection[0].attribute == {"label": "cat"}


def test_image_collection_add_tar_reads_and_copies_images(tmp_path: Path):
    png = _png_bytes()
    tar_path = tmp_path / "images.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        _add_tar_member(tar, "a.png", png)
        _add_tar_member(tar, "a.json", json.dumps({"label": "cat"}).encode())

    collection = ImageCollection()
    collection.add_tar(tar_path)

    assert collection[0].to_pil().size == (2, 3)

    destination = tmp_path / "dst"
    collection.save(destination, images_per_dir=None)

    assert (destination / "a.png").read_bytes() == png
    assert json.loads((destination / "a.json").read_text()) == {"label": "cat"}


def test_image_collection_add_attributes_and_override():
    collection = ImageCollection()
    collection.add("a.jpg", {"label": "original"})
    collection.add("b.png")

    collection.add_attributes(
        [
            ({"path": "a.jpg"}, {"label": "updated"}),
            ({"path": "b.png"}, {"label": "new"}),
        ]
    )
    assert collection[0].attribute == {"label": "original"}
    assert collection[1].attribute == {"label": "new"}

    collection.add_attributes(
        [
            ({"path": "a.jpg"}, {"label": "updated"}),
            ({"path": "missing.jpg"}, {"label": "ignored"}),
        ],
        override_existing_attributes=True,
    )
    assert collection[0].attribute == {"label": "updated"}
    assert len(collection) == 2


def test_image_collection_add_attributes_to_tar_images(tmp_path: Path):
    tar_path = tmp_path / "images.tar"
    with tarfile.open(tar_path, "w") as tar:
        _add_tar_member(tar, "a.png", b"a")
        _add_tar_member(tar, "b.jpg", b"b")

    collection = ImageCollection()
    collection.add_tar(tar_path)

    collection.add_attributes(
        [
            ({"tar": str(tar_path), "member": "a.png"}, {"label": "cat"}),
            ({"tar": str(tar_path), "member": "b.jpg"}, {"label": "dog"}),
            ({"tar": str(tar_path), "member": "missing.png"}, {"label": "ignored"}),
        ]
    )

    assert {image.source.path_in_tar: image.attribute for image in collection} == {
        "a.png": {"label": "cat"},
        "b.jpg": {"label": "dog"},
    }


def test_image_collection_add_attributes_tar_and_override(tmp_path: Path):
    tar_path = tmp_path / "images.tar"
    with tarfile.open(tar_path, "w") as tar:
        _add_tar_member(tar, "a.png", b"a")
        _add_tar_member(tar, "a.json", json.dumps({"label": "original"}).encode())
        _add_tar_member(tar, "b.jpg", b"b")

    collection = ImageCollection()
    collection.add_tar(tar_path)

    collection.add_attributes(
        [
            ({"tar": str(tar_path), "member": "a.png"}, {"label": "updated"}),
            ({"tar": str(tar_path), "member": "b.jpg"}, {"label": "new"}),
        ]
    )
    assert next(
        image for image in collection if image.source.path_in_tar == "a.png"
    ).attribute == {"label": "original"}
    assert next(
        image for image in collection if image.source.path_in_tar == "b.jpg"
    ).attribute == {"label": "new"}

    collection.add_attributes(
        [
            ({"tar": str(tar_path), "member": "a.png"}, {"label": "updated"}),
            ({"tar": str(tar_path), "member": "missing.png"}, {"label": "ignored"}),
        ],
        override_existing_attributes=True,
    )
    assert next(
        image for image in collection if image.source.path_in_tar == "a.png"
    ).attribute == {"label": "updated"}


def test_image_collection_add_attributes_distinguishes_same_member_in_tars(
    tmp_path: Path,
):
    first_tar_path = tmp_path / "first.tar"
    second_tar_path = tmp_path / "second.tar"
    for tar_path in (first_tar_path, second_tar_path):
        with tarfile.open(tar_path, "w") as tar:
            _add_tar_member(tar, "00000.jpg", b"image")

    collection = ImageCollection()
    collection.add_tar(first_tar_path)
    collection.add_tar(second_tar_path)

    collection.add_attributes(
        [
            (
                {"tar": str(first_tar_path), "member": "00000.jpg"},
                {"source": "first"},
            ),
            (
                {"tar": str(second_tar_path), "member": "00000.jpg"},
                {"source": "second"},
            ),
        ]
    )

    assert [image.attribute for image in collection] == [
        {"source": "first"},
        {"source": "second"},
    ]


def test_image_collection_add_attributes_rejects_duplicate_keys(tmp_path: Path):
    tar_path = tmp_path / "images.tar"
    with tarfile.open(tar_path, "w") as tar:
        _add_tar_member(tar, "a.png", b"a")

    collection = ImageCollection()
    collection.add_tar(tar_path)

    with pytest.raises(
        RuntimeWarning, match="Multiple attributes mapped to single key"
    ):
        collection.add_attributes(
            [
                ({"tar": str(tar_path), "member": "a.png"}, {"label": "first"}),
                ({"tar": str(tar_path), "member": "a.png"}, {"label": "second"}),
            ]
        )

    assert collection[0].attribute is None


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
