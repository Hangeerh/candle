from candle.image import ImageCollection


def test_image_collection():
    col = ImageCollection()
    col.add("./test_image.py")
    assert col.images == []
