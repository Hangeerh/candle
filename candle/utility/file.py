from pathlib import Path


def _norm_path(path: str | Path):
    if not isinstance(path, Path):
        return Path(path)
    return path


def _is_macos_metadata(path: str | Path) -> bool:
    """Return True for macOS metadata entries such as AppleDouble files.

    macOS archives can contain ``._*`` resource-fork files alongside real files.
    Tar archives may also store them under a ``__MACOSX`` directory. These entries
    are not valid images or JSON documents and should not be treated as samples.
    """
    parts = _norm_path(path).parts
    return "__MACOSX" in parts or any(part.startswith("._") for part in parts)


def _is_valid_img_extension(image: str | Path) -> bool:
    extens = _norm_path(image).suffix
    return extens == ".jpg" or extens == ".png" or extens == ".jpeg"
