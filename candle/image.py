from pathlib import Path

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

    def save(self, dir: str | Path):
        ndir = _norm_path(dir)
        ndir.mkdir(parents=True, exist_ok=True)

        for img in self.images:
            src = img.dir / img.file
            if img.new_name is None:
                dst = ndir / img.file
            else:
                dst = ndir / img.new_name
            src.copy(dst, preserve_metadata=True)


class Image:
    def __init__(self, image: Path):
        path = _norm_path(image)
        self.file = path.name
        self.dir = path.parent
        self.new_name = None


def _norm_path(path: str | Path):
    if not isinstance(path, Path):
        return Path(path)
    return path


def _is_valid_extension(image: str | Path) -> bool:
    extens = _norm_path(image).suffix
    return extens == ".jpg" or extens == ".png" or extens == ".jpeg"
