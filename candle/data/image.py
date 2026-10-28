import tarfile
import shutil
import torchvision
import io
from pathlib import Path
from collections.abc import Callable
from collections import defaultdict
import PIL.Image
import PIL.ImageFile
import torch
from typing import Literal
from torchvision.io import read_image
import json

__all__ = ["ImageCollection", "Image", "ImageSourceFile", "ImageSourceTar"]


class ImageCollection:
    def __init__(self):
        self.images: list[Image] = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

    def add(self, image: str | Path, attribute: dict | None = None):
        """Add one image to the ImageCollection

        Args:
            image: Path to the image to be added.
            attribute: Attributes attached to the image, for example labels, captions, etc.
        """
        path = _norm_path(image)

        if not _is_valid_img_extension(path):
            return

        self.images.append(Image(ImageSourceFile(path), attribute))

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

    def add_tar(self, tar: str | Path):
        """Add all images in a tarball. Supports WebDataset format.

        Args:
            tar: The path to the tarball containing images.
        """
        path = _norm_path(tar)

        if not path.is_file():
            raise RuntimeError("Provided path is not a file")

        # Group members by their name without the extension, just like add_dir.
        # Read the archive sequentially once instead of looking up every image
        # by name, which would scan the member list repeatedly.
        valid_extens = (".jpg", ".jpeg", ".png")
        valid_suffixes = {*valid_extens, ".json"}

        grouped_samples: dict[str, dict[str, tarfile.TarInfo]] = defaultdict(dict)
        with tarfile.open(path, "r") as tar_file:
            for member in tar_file:
                if not member.isfile():
                    continue

                member_path = Path(member.name)
                ext = member_path.suffix.lower()
                if ext not in valid_suffixes:
                    continue

                grouped_samples[str(member_path.with_suffix(""))][ext] = member

            for extens in grouped_samples.values():
                img_ext = next((ext for ext in extens if ext in valid_extens), None)
                if img_ext is None:
                    continue

                attributes = None
                json_member = extens.get(".json")
                if json_member is not None:
                    json_file = tar_file.extractfile(json_member)
                    if json_file is not None:
                        attributes = json.load(json_file)

                image_member = extens[img_ext]
                self.images.append(
                    Image(
                        ImageSourceTar(path, image_member.name, image_member),
                        attributes,
                    )
                )

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
                if image.source.file_name + image.source.file_extens == image_file:
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
            scheme: "numbered": Name images by number, from 0 to number of images.
                                For example, 000.jpg, 002.png, 003.jpeg.
        """
        if scheme == "numbered":
            digits = len(str(abs(len(self.images))))
            for index, img in enumerate(self.images):
                if img.keep:
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
                    img.copy_to_dir(dir)

        kept_images = [img for img in self.images if img.keep]
        if images_per_dir is not None:
            assert images_per_dir > 0, "images_per_dir must be greater than 0"

            # This does ceiling division to include the remainder in the chunks.
            for i in range(-(-len(self.images) // images_per_dir)):
                dst_dir = ndir / f"{i:05d}"
                dst_dir.mkdir(exist_ok=True)

                copy_images_to_dir(
                    kept_images[images_per_dir * i : (i + 1) * images_per_dir], dst_dir
                )

            return

        copy_images_to_dir(kept_images, ndir)


class ImageSourceFile:
    def __init__(self, path: str | Path):
        path = _norm_path(path)
        self.file_name = path.stem
        self.file_extens = path.suffix.lower()
        self.file_dir = path.parent

    def to_tensor(self) -> torch.Tensor:
        return read_image(str(self.file_dir) + "/" + self.file_name + self.file_extens)

    def to_pil(self) -> PIL.ImageFile.ImageFile:
        return PIL.Image.open(
            str(self.file_dir) + "/" + self.file_name + self.file_extens
        )

    def write_to_file(self, file: Path):
        src = self.file_dir / (self.file_name + self.file_extens)
        src.copy(file, preserve_metadata=True)


class ImageSourceTar:
    persist_file_handles: bool = False
    open_tar_handles: dict[str, tarfile.TarFile] = {}

    def __init__(
        self,
        tarfile: str | Path,
        path_in_tar: str,
        member: tarfile.TarInfo | None = None,
    ):
        self.tarfile = str(_norm_path(tarfile))
        path = _norm_path(path_in_tar)
        self.path_in_tar = str(path)
        self.file_name = path.stem
        self.file_extens = path.suffix.lower()
        self.file_dir = path.parent
        # Reusing the TarInfo avoids searching the archive's member list when
        # reading the image later.
        self._member = member

    def to_tensor(self) -> torch.Tensor:
        return torchvision.transforms.ToPILImage()(self.to_pil())

    def to_pil(self) -> PIL.ImageFile.ImageFile:
        handle = ImageSourceTar.open_tar_handles.get(self.tarfile)

        handle_does_not_exist = handle is None
        if handle_does_not_exist:
            handle = tarfile.open(self.tarfile, "r")

        member = self._member if self._member is not None else self.path_in_tar
        image = handle.extractfile(member)

        if image is None:
            raise FileNotFoundError(
                f"Could not find tar entry {self.path_in_tar} in {self.tarfile}"
            )

        image = image.read()
        image = PIL.Image.open(io.BytesIO(image))

        if ImageSourceTar.persist_file_handles:
            if handle_does_not_exist:
                ImageSourceTar.open_tar_handles[self.tarfile] = handle
        else:
            handle.close()

        return image

    def write_to_file(self, file: Path):
        handle = ImageSourceTar.open_tar_handles.get(self.tarfile)

        handle_does_not_exist = handle is None
        if handle_does_not_exist:
            handle = tarfile.open(self.tarfile, "r")

        member = self._member if self._member is not None else self.path_in_tar
        image_file = handle.extractfile(member)
        if image_file is None:
            raise FileNotFoundError(
                f"Could not find tar entry {self.path_in_tar} in {self.tarfile}"
            )

        with open(file, "wb") as f:
            shutil.copyfileobj(image_file, f)

        if ImageSourceTar.persist_file_handles:
            if handle_does_not_exist:
                ImageSourceTar.open_tar_handles[self.tarfile] = handle
        else:
            handle.close()

    @classmethod
    def clean_tar_handles(cls):
        for handle in ImageSourceTar.open_tar_handles.values():
            handle.close()
        ImageSourceTar.open_tar_handles = {}


ImageSource = ImageSourceFile | ImageSourceTar


class Image:
    def __init__(self, source: ImageSource, attribute: dict | None = None):
        self.source = source
        self.new_name: str | None = None
        self.attribute: dict | None = attribute
        self.keep = True

    def copy_to_dir(self, dir: str | Path):
        dir = _norm_path(dir)

        name = self.source.file_name
        if self.new_name is not None:
            name = self.new_name
        dst_image = dir / (name + self.source.file_extens)

        if not self.attribute is None:
            with open(dir / (name + ".json"), "w") as f:
                json.dump(self.attribute, f)

        self.source.write_to_file(dst_image)

    def to_tensor(self) -> torch.Tensor:
        """Return the image as a torch tensor"""
        return self.source.to_tensor()

    def to_pil(self) -> PIL.ImageFile.ImageFile:
        """Return the image as a PIL image"""
        return self.source.to_pil()

    def show(self):
        """Show the image using PIL"""
        self.to_pil().show()


def _norm_path(path: str | Path):
    if not isinstance(path, Path):
        return Path(path)
    return path


def _is_valid_img_extension(image: str | Path) -> bool:
    extens = _norm_path(image).suffix
    return extens == ".jpg" or extens == ".png" or extens == ".jpeg"
