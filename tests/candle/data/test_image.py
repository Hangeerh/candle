import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image as PILImage

from candle.data.image import ImageCollection, ImageSourceFile, ImageSourceTar
from candle.data.utility import (
    _is_macos_metadata,
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


def test_add():
    collection = ImageCollection()
    assert len(collection) == 0

    collection.add("photo.jpg", {"label": "cat"})
    collection.add(Path("photo.png"))
    collection.add("photo.jpeg", {"label": "dog"})
    collection.add("photo.gif")
    collection.add("photo")
    collection.add("._photo.jpg")
    collection.add("__MACOSX/photo.jpg")

    assert len(collection) == 3
    assert all(isinstance(image.source, ImageSourceFile) for image in collection)
    assert [
        image.source.file_name + image.source.file_extens for image in collection
    ] == [
        "photo.jpg",
        "photo.png",
        "photo.jpeg",
    ]
    assert [image.attribute for image in collection] == [
        {"label": "cat"},
        None,
        {"label": "dog"},
    ]
    assert all(image.keep for image in collection)
    assert all(image.new_name is None for image in collection)

    assert collection[0] is collection.images[0]
    assert collection[-1] is collection.images[-1]
    assert collection[:2] == collection.images[:2]


def test_add_dir(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Provided path is not a directory"):
        ImageCollection().add_dir(tmp_path / "missing")

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "a.jpg").write_bytes(b"a")
    (source_dir / "a.json").write_text(json.dumps({"label": "cat"}))
    (source_dir / "b.png").write_bytes(b"b")
    (source_dir / "c.txt").write_text("ignored")
    (source_dir / "orphan.json").write_text(json.dumps({"label": "orphan"}))
    (source_dir / "._a.jpg").write_bytes(b"metadata")
    (source_dir / "._a.json").write_text("not json")

    macos_dir = source_dir / "__MACOSX"
    macos_dir.mkdir()
    (macos_dir / "._b.png").write_bytes(b"metadata")

    nested_dir = source_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "nested.jpeg").write_bytes(b"nested")
    (nested_dir / "nested.json").write_text(json.dumps({"label": "nested"}))

    flat = ImageCollection()
    flat.add_dir(source_dir)

    assert len(flat) == 2
    assert {image.source.file_name + image.source.file_extens for image in flat} == {
        "a.jpg",
        "b.png",
    }
    assert {image.source.file_name: image.attribute for image in flat} == {
        "a": {"label": "cat"},
        "b": None,
    }

    recursive = ImageCollection()
    recursive.add_dir(source_dir, recursive=True)

    assert len(recursive) == 3
    assert {
        image.source.file_name + image.source.file_extens for image in recursive
    } == {
        "a.jpg",
        "b.png",
        "nested.jpeg",
    }
    assert {image.source.file_name: image.attribute for image in recursive} == {
        "a": {"label": "cat"},
        "b": None,
        "nested": {"label": "nested"},
    }


def test_add_tar(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Provided path is not a file"):
        ImageCollection().add_tar(tmp_path / "missing.tar")

    png = _png_bytes()
    tar_path = tmp_path / "images.tar.gz"
    tar_members = {
        "a.png": png,
        "a.json": json.dumps({"label": "cat"}).encode(),
        "b.jpg": png,
        "c.txt": b"ignored",
        "orphan.json": json.dumps({"label": "orphan"}).encode(),
        "._a.png": b"metadata",
        "__MACOSX/00000/._b.jpg": b"metadata",
        "nested/d.jpeg": png,
        "nested/d.json": json.dumps({"label": "dog"}).encode(),
    }
    with tarfile.open(tar_path, "w:gz") as tar:
        for name, data in tar_members.items():
            _add_tar_member(tar, name, data)

    collection = ImageCollection()
    collection.add_tar(tar_path)

    assert len(collection) == 3
    assert all(isinstance(image.source, ImageSourceTar) for image in collection)
    assert {image.source.path_in_tar for image in collection} == {
        "a.png",
        "b.jpg",
        "nested/d.jpeg",
    }
    assert {
        image.source.file_name + image.source.file_extens for image in collection
    } == {
        "a.png",
        "b.jpg",
        "d.jpeg",
    }
    assert {image.source.path_in_tar: image.attribute for image in collection} == {
        "a.png": {"label": "cat"},
        "b.jpg": None,
        "nested/d.jpeg": {"label": "dog"},
    }

    first = next(image for image in collection if image.source.path_in_tar == "a.png")
    assert first.to_pil().size == (2, 3)
    assert first.source.open().read() == png


def test_add_attributes(tmp_path: Path):
    files = ImageCollection()
    files.add("a.jpg", {"label": "original"})
    files.add("b.png")
    files.add("c.jpeg")

    files.add_attributes(
        [
            ({"path": "a.jpg"}, {"label": "updated"}),
            ({"path": "b.png"}, {"label": "new"}),
            ({"path": "missing.jpg"}, {"label": "ignored"}),
        ]
    )
    assert [image.attribute for image in files] == [
        {"label": "original"},
        {"label": "new"},
        None,
    ]

    files.add_attributes(
        [
            ({"path": "a.jpg"}, {"label": "updated"}),
            ({"path": "b.png"}, {"label": "also updated"}),
            ({"path": "missing.jpg"}, {"label": "ignored"}),
        ],
        override_existing_attributes=True,
    )
    assert [image.attribute for image in files] == [
        {"label": "updated"},
        {"label": "also updated"},
        None,
    ]

    first_tar = tmp_path / "first.tar"
    first_tar_members = {
        "a.png": b"a",
        "a.json": json.dumps({"label": "original"}).encode(),
        "b.jpg": b"b",
    }
    with tarfile.open(first_tar, "w") as tar:
        for name, data in first_tar_members.items():
            _add_tar_member(tar, name, data)

    second_tar = tmp_path / "second.tar"
    with tarfile.open(second_tar, "w") as tar:
        _add_tar_member(tar, "a.png", b"second")

    tars = ImageCollection()
    tars.add_tar(first_tar)
    tars.add_tar(second_tar)
    tars.add_attributes(
        [
            ({"tar": str(first_tar), "member": "a.png"}, {"label": "updated"}),
            ({"tar": str(first_tar), "member": "b.jpg"}, {"label": "new"}),
            ({"tar": str(second_tar), "member": "a.png"}, {"label": "second"}),
            ({"tar": str(first_tar), "member": "missing.png"}, {"label": "ignored"}),
        ]
    )
    assert {
        (image.source.tarfile, image.source.path_in_tar): image.attribute
        for image in tars
    } == {
        (str(first_tar), "a.png"): {"label": "original"},
        (str(first_tar), "b.jpg"): {"label": "new"},
        (str(second_tar), "a.png"): {"label": "second"},
    }

    tars.add_attributes(
        [({"tar": str(first_tar), "member": "a.png"}, {"label": "updated"})],
        override_existing_attributes=True,
    )
    assert next(
        image
        for image in tars
        if image.source.tarfile == str(first_tar)
        and image.source.path_in_tar == "a.png"
    ).attribute == {"label": "updated"}

    duplicates = ImageCollection()
    duplicates.add("duplicate.jpg")
    with pytest.raises(
        RuntimeWarning, match="Multiple attributes mapped to single key"
    ):
        duplicates.add_attributes(
            [
                ({"path": "duplicate.jpg"}, {"label": "first"}),
                ({"path": "duplicate.jpg"}, {"label": "second"}),
            ]
        )
    assert duplicates[0].attribute is None


def test_filter():
    collection = ImageCollection()
    collection.add("first.jpg")
    collection.add("second.jpg")
    collection.add("third.jpg")

    collection.filter(lambda image: image.source.file_name != "second")
    assert [image.keep for image in collection] == [True, False, True]

    collection.filter(lambda image: image.source.file_name == "third")
    assert [image.keep for image in collection] == [False, False, True]


def test_merge():
    first = ImageCollection()
    first.add("first.jpg")
    first_image = first[0]
    second = ImageCollection()
    second.add("second.jpg")
    third = ImageCollection()
    third.add("third.jpg")

    first.merge(second)
    assert first.images == [first_image, second[0]]

    first.merge([third])
    assert first.images == [first_image, second[0], third[0]]

    first.merge([])
    assert len(first) == 3


def test_batch_rename():
    collection = ImageCollection()
    for index in range(12):
        collection.add(f"{index}.jpg")

    collection.filter(lambda image: int(image.source.file_name) % 5 == 0)
    collection.batch_rename()

    new_names = {image.source.file_name: image.new_name for image in collection}
    assert new_names["0"] == "00"
    assert new_names["5"] == "05"
    assert new_names["10"] == "10"
    assert new_names["1"] is None
    assert new_names["11"] is None


def test_save(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    colors = {
        "first": (255, 0, 0),
        "second": (0, 255, 0),
        "third": (0, 0, 255),
        "fourth": (255, 255, 0),
    }
    for name, color in colors.items():
        PILImage.new("RGB", (2, 3), color).save(source_dir / f"{name}.png")

    collection = ImageCollection()
    for name in colors:
        collection.add(source_dir / f"{name}.png")
    collection.add_attributes(
        [({"path": str(source_dir / "first.png")}, {"label": "first"})]
    )
    collection.filter(lambda image: image.source.file_name != "second")

    with pytest.raises(
        ValueError, match="Cannot create shards with less than 1 element"
    ):
        collection.save(tmp_path / "invalid", images_per_shard=0)

    flat = tmp_path / "flat"
    collection.save(flat, images_per_shard=None)
    assert sorted(path.name for path in flat.iterdir()) == [
        "first.json",
        "first.png",
        "fourth.png",
        "third.png",
    ]
    assert (flat / "first.png").read_bytes() == (source_dir / "first.png").read_bytes()
    assert json.loads((flat / "first.json").read_text()) == {"label": "first"}

    sharded = tmp_path / "sharded"
    collection.save(sharded, images_per_shard=2)
    assert sorted(path.name for path in sharded.iterdir()) == ["00000", "00001"]
    assert sorted(path.name for path in (sharded / "00000").iterdir()) == [
        "first.json",
        "first.png",
        "third.png",
    ]
    assert sorted(path.name for path in (sharded / "00001").iterdir()) == ["fourth.png"]

    collection.batch_rename()
    assert [(image.source.file_name, image.new_name) for image in collection] == [
        ("first", "0"),
        ("second", None),
        ("third", "2"),
        ("fourth", "3"),
    ]

    tar_flat = tmp_path / "tar-flat"
    collection.save(tar_flat, images_per_shard=None, tar=True)
    assert [path.name for path in tar_flat.iterdir()] == ["00000.tar"]
    with tarfile.open(tar_flat / "00000.tar") as tar:
        members = {
            member.name: tar.extractfile(member).read()
            for member in tar
            if member.isfile()
        }
    assert set(members) == {"0.png", "2.png", "3.png"}
    assert members["0.png"] == (source_dir / "first.png").read_bytes()
    assert members["2.png"] == (source_dir / "third.png").read_bytes()
    assert members["3.png"] == (source_dir / "fourth.png").read_bytes()

    tar_sharded = tmp_path / "tar-sharded"
    collection.save(tar_sharded, images_per_shard=2, tar=True)
    assert sorted(path.name for path in tar_sharded.iterdir()) == [
        "00000.tar",
        "00001.tar",
    ]
    with tarfile.open(tar_sharded / "00000.tar") as tar:
        assert [member.name for member in tar if member.isfile()] == ["0.png", "2.png"]
    with tarfile.open(tar_sharded / "00001.tar") as tar:
        assert [member.name for member in tar if member.isfile()] == ["3.png"]

    png = _png_bytes()
    source_tar = tmp_path / "source.tar"
    source_tar_members = {
        "a.png": png,
        "a.json": json.dumps({"label": "a"}).encode(),
        "b.png": png,
        "c.png": png,
    }
    with tarfile.open(source_tar, "w") as tar:
        for name, data in source_tar_members.items():
            _add_tar_member(tar, name, data)
    tar_collection = ImageCollection()
    tar_collection.add_tar(source_tar)

    tar_source_dir = tmp_path / "tar-source"
    tar_collection.save(tar_source_dir, images_per_shard=None)
    assert sorted(path.name for path in tar_source_dir.iterdir()) == [
        "a.json",
        "a.png",
        "b.png",
        "c.png",
    ]
    assert json.loads((tar_source_dir / "a.json").read_text()) == {"label": "a"}
    for name in ("a", "b", "c"):
        assert (tar_source_dir / f"{name}.png").read_bytes() == png
