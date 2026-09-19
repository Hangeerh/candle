from pathlib import Path
from collections.abc import Callable
from collections import defaultdict
import PIL.Image
import PIL.ImageFile
import torch
from typing import Literal
from torchvision.io import read_image
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

        if not _is_valid_img_extension(path):
            return

        self.images.append(Image(path, attribute))

    def add_dir(
        self,
        dir: str | Path,
        recursive: bool = False,
    ):
        """Add all images in a directory. Supports WebDataset format.

        Args:
            dir: The directory to add.
            recursive: Recurses subdirectories if True, don't if False.
        """
        path = _norm_path(dir)

        if not path.is_dir():
            raise RuntimeError("Provided path is not a directory")

        grouped_samples = defaultdict(dict)
        pattern = "**/*" if recursive else "*"
        for file in path.glob(pattern):
            if file.is_file():
                prefix = file.stem
                ext = file.suffix.lower()
                grouped_samples[prefix][ext] = file

        valid_extens = [".jpg", ".jpeg", ".png"]

        for prefix, extens in grouped_samples.items():
            img_ext = next((ext for ext in extens if ext in valid_extens), None)

            if img_ext:
                img_file = extens[img_ext]

                attributes = None
                if ".json" in extens:
                    with open(extens[".json"], "r") as f:
                        attributes = json.load(f)

                self.add(img_file, attributes)

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
        for image_file, attribute in attribute_list:
            for image in self.images:
                if image.file_name + image.file_extens == image_file:
                    if image.attribute is None or override_existing_attributes:
                        image.attribute = attribute
                    break

    def filter(self, filter_func: Callable[[Image], bool]):
        """Filter images with the given function.

        Args:
            filter_func: A function f(Image) -> bool that determines which images to
                keep. Should return True to keep image, and False to discard image.
        """

        for img in self.images:
            img.keep = filter_func(img)

    def merge(self, collections: list[ImageCollection] | ImageCollection):
        """Merge multiple collections into the current ImageCollection.

        Args:
            collections: List of ImageCollections to merge.
        """

        if isinstance(collections, ImageCollection):
            self.images.extend(collections.images)
            return

        for col in collections:
            self.images.extend(col.images)

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

    def save(self, dir: str | Path, images_per_dir: None | int):
        """Save the contents of the ImageCollection to a directory. The saved data will be
        in WebDataset format.

        Args:
            dir: The directory to save images.
            images_per_dir: If set to an int, shard the collection into multiple subdirs
                in the given directory. Each subdir contains number of images specified
                by images_per_dir. When set to None, everything will be put into one dir.
        """
        ndir = _norm_path(dir)
        ndir.mkdir(parents=True, exist_ok=True)

        def copy_images_to_dir(chunk: list[Image], dir: Path):
            for img in chunk:
                if img.keep:
                    src = img.file_dir / (img.file_name + img.file_extens)

                    name = img.file_name
                    if img.new_name is not None:
                        name = img.new_name
                    dst_image = dir / (name + img.file_extens)

                    if not img.attribute is None:
                        dst_meta = dir / (name + ".json")
                        with open(dst_meta, "w") as f:
                            json.dump(img.attribute, f)

                    src.copy(dst_image, preserve_metadata=True)

        if images_per_dir is not None:
            assert images_per_dir > 0, "images_per_dir must be greater than 0"

            # This does ceiling division to include the remainder in the chunks.
            for i in range(-(-len(self.images) // images_per_dir)):
                dst_dir = ndir / f"{i:05d}"
                dst_dir.mkdir()

                copy_images_to_dir(
                    self.images[images_per_dir * i : (i + 1) * images_per_dir], dst_dir
                )

            return

        copy_images_to_dir(self.images, ndir)


class Image:
    def __init__(self, image: str | Path, attribute: dict | None = None):
        path = _norm_path(image)
        self.file_name: str = path.stem
        self.file_extens: str = path.suffix.lower()
        self.file_dir: Path = path.parent
        self.new_name: str | None = None
        self.attribute: dict | None = attribute
        self.keep = True

    def to_tensor(self) -> torch.Tensor:
        return read_image(str(self.file_dir) + "/" + self.file_name + self.file_extens)

    def to_pil(self) -> PIL.ImageFile.ImageFile:
        return PIL.Image.open(
            str(self.file_dir) + "/" + self.file_name + self.file_extens
        )


def _norm_path(path: str | Path):
    if not isinstance(path, Path):
        return Path(path)
    return path


def _is_valid_img_extension(image: str | Path) -> bool:
    extens = _norm_path(image).suffix
    return extens == ".jpg" or extens == ".png" or extens == ".jpeg"
