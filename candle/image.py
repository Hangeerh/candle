from pathlib import Path


class ImageCollection:
    def __init__(self):
        self.images: list[Path] = []

    def add(self, image: str | Path):
        path = self._norm_path(image)

        if not self._is_valid_extension(path):
            return

        self.images.append(Path(path))

    def add_dir(self, dir: str | Path, recursive: bool = False):
        path = self._norm_path(dir)

        if not path.is_dir():
            raise RuntimeError("Provided path is not a directory")

        for root, dirs, files in path.walk():
            for file in files:
                self.add(root / file)
            if recursive:
                for dir in dirs:
                    self.add_dir(root / dir)

    def _norm_path(self, path: str | Path) -> Path:
        if not isinstance(path, Path):
            return Path(path)
        return path

    def _is_valid_extension(self, image: str | Path) -> bool:
        extens = self._norm_path(image).suffix
        return extens == ".jpg" or extens == ".png" or extens == ".jpeg"
