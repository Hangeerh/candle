from pathlib import Path
from typing import Literal

__all__ = ["ImageCollection"]


class ImageCollection:
    def __init__(self):
        self.images: list[Image] = []

    def add(self, image: str | Path):
        path = _norm_path(image)

        if not _is_valid_extension(path):
            return

        self.images.append(Image(path))

    def add_dir(self, dir: str | Path, recursive: bool = False):
        path = _norm_path(dir)

        if not path.is_dir():
            raise RuntimeError("Provided path is not a directory")

        pattern = "**/*" if recursive else "*"
        for file in path.glob(pattern):
            if file.is_file():
                self.add(file)

    def batch_rename(self, scheme: Literal["numbered"] = "numbered"):
        if scheme == "numbered":
            digits = len(str(abs(len(self.images))))
            for index, img in enumerate(self.images):
                img.new_name = f"{index:0{digits}d}"

    def save(self, dir: str | Path):
        ndir = _norm_path(dir)
        ndir.mkdir(parents=True, exist_ok=True)

        for img in self.images:
            src = img.dir / (img.name + img.extens)
            if img.new_name is None:
                dst = ndir / (img.name + img.extens)
            else:
                dst = ndir / (img.new_name + img.extens)
            src.copy(dst, preserve_metadata=True)


class Image:
    def __init__(self, image: Path):
        path = _norm_path(image)
        self.name: str = path.stem
        self.extens: str = path.suffix
        self.dir: Path = path.parent
        self.new_name: str | None = None


def _norm_path(path: str | Path):
    if not isinstance(path, Path):
        return Path(path)
    return path


def _is_valid_extension(image: str | Path) -> bool:
    extens = _norm_path(image).suffix
    return extens == ".jpg" or extens == ".png" or extens == ".jpeg"
