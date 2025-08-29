from PIL import Image
import numpy as np
import polars as pl


class ImageLoader:
    @staticmethod
    def load_image(filename: str, *, is_1_dim: bool = False) -> np.ndarray:
        img = Image.open(filename).resize((128, 128))
        img.load()
        img = np.asarray(img, dtype=np.int32)

        return img.reshape((-1, 3)) if is_1_dim else img

    @staticmethod
    def save_image(data: np.ndarray, filename: str) -> None:
        img = Image.fromarray(np.asarray(np.clip(data, 0, 255), dtype=np.uint8))
        img.save(fp=filename)

    @staticmethod
    def img_to_df(data: np.ndarray):
        return pl.DataFrame(
            data,
            schema={'r': pl.UInt8, 'g': pl.UInt8, 'b': pl.UInt8}
        )
