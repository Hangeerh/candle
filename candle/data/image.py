from pathlib import Path
from typing import Literal
import json

__all__ = ["ImageCollection", "Image"]


class ImageCollection:
    def __init__(self):
        self.images: list[Image] = []

    def add(self, image: str | Path, attribute: dict | None = None):
        """Add one image to the ImageCollection

        Args:
            image: Path to the image to be added.
            attribute: Attributes attached to the image, for example labels, captions, etc.
        """
        path = _norm_path(image)

        if not _is_valid_extension(path):
            return

        self.images.append(Image(path, attribute))

    def add_dir(
        self,
        dir: str | Path,
        recursive: bool = False,
    ):
        """Add all images in a directory.

        Args:
            dir: The directory to add.
            recursive: Recurese subdirectories if True, don't if False.
        """
        path = _norm_path(dir)

        if not path.is_dir():
            raise RuntimeError("Provided path is not a directory")

        pattern = "**/*" if recursive else "*"
        for file in path.glob(pattern):
            if file.is_file():
                self.add(file)

    def add_attributes(
        self,
        attribute_list: list[tuple[str, dict]],
        override_existing_attributes: bool = False,
    ):
        """Attaches attributes to images in the ImageCollection.

        Args:
            attribute_list: List of (image_name, attribute_dict) tuples. Note that
                image_name does not include the extension.
            override_existing_attributes: If True, the newly specified attribute
                will override existing attributes.
        """
        for image_name, attribute in attribute_list:
            for image in self.images:
                if image.name == image_name:
                    if image.attribute is None or override_existing_attributes:
                        image.attribute = attribute
                    break

    def batch_rename(self, scheme: Literal["numbered"] = "numbered"):
        """Rename all Images according to the specified scheme.

        Args:
            scheme: "numbered": Name images by number, from 1 to number of images.
                                For example, 001.jpg, 002.png, 003.jpeg.
        """
        if scheme == "numbered":
            digits = len(str(abs(len(self.images))))
            for index, img in enumerate(self.images):
                img.new_name = f"{index:0{digits}d}"

    def save(self, dir: str | Path):
        ndir = _norm_path(dir)
        ndir.mkdir(parents=True, exist_ok=True)

        image_attribute_json: list[dict[str, dict]] = []

        for img in self.images:
            src = img.dir / (img.name + img.extens)

            name = img.name
            if img.new_name is not None:
                name = img.new_name
            dst = ndir / (name + img.extens)

            if not img.attribute is None:
                image_attribute_json.append({img.name: img.attribute})

            src.copy(dst, preserve_metadata=True)

        if image_attribute_json:
            with open(ndir / "meta.json", "w") as f:
                json.dump(image_attribute_json, f, indent=2)


class Image:
    def __init__(self, image: Path, attribute: dict | None = None):
        path = _norm_path(image)
        self.name: str = path.stem
        self.extens: str = path.suffix
        self.dir: Path = path.parent
        self.new_name: str | None = None
        self.attribute: dict | None = attribute


def _norm_path(path: str | Path):
    if not isinstance(path, Path):
        return Path(path)
    return path


def _is_valid_extension(image: str | Path) -> bool:
    extens = _norm_path(image).suffix
    return extens == ".jpg" or extens == ".png" or extens == ".jpeg"
