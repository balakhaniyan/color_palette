import math
from enum import Enum
from itertools import count
from tqdm import tqdm
import polars as pl

from LAB import Lab
from image_loader import ImageLoader


class Threshold(float, Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3.5
    VERY_LOW = 5


class Palette:

    def __init__(self, *, threshold=5, sum_of_percentage=80, number_of_most=3):
        self.threshold = threshold
        self.sum_of_percentage = sum_of_percentage
        self.number_of_most = number_of_most
        self.colors: pl.DataFrame | None = None

    def uniqueness(self) -> pl.DataFrame:
        return (
            self.colors
            .group_by('hex')
            .agg(
                pl.col('L').first(),
                pl.col('A').first(),
                pl.col('B').first(),
                pl.len().alias('percentage') / self.colors.shape[0] * 100,
            )
        )

    def get_most_colors(self, filename: str):
        image = ImageLoader.img_to_df(ImageLoader.load_image(filename, is_1_dim=True))

        image = Lab.rgb_to_lab(image)

        self.colors: pl.DataFrame = image
        self.colors = self.uniqueness().sort('percentage')

        for _ in tqdm(count()):
            shape = self.colors.shape[0]
            if shape > 1500:
                sample_size = 1000
            elif shape > 200:
                sample_size = int(shape * 0.8)
            elif shape > 50:
                sample_size = int(shape * .4)
            elif shape > 10:
                sample_size = int(shape * .1)
            else:
                sample_size = 2
            
            sample_colors = self.colors.sample(sample_size)
            other_colors = self.colors.join(sample_colors, on='hex', how='anti')

            joined_colors = sample_colors.join(other_colors, how='cross', suffix='_2')

            with_delta_e = (
                joined_colors
                .with_columns(
                    delta_e=
                    ((pl.col('L') - pl.col('L_2')) ** 2 +
                     (pl.col('A') - pl.col('A_2')) ** 2 +
                     (pl.col('B') - pl.col('B_2')) ** 2) ** (1 / 2)
                )
                .filter(pl.col('delta_e') < self.threshold)
                .sort(['hex', 'delta_e'])
                .group_by('hex', maintain_order=True)
                .agg(pl.all().first())
            )

            more_than_one_occurrence = with_delta_e.select('hex_2').group_by('hex_2').count().filter(
                pl.col('count') > 1)

            with_delta_e = with_delta_e.join(more_than_one_occurrence, on='hex_2', how='anti')

            sum_of_percentages = with_delta_e.select(
                pl.when(pl.col('percentage') > pl.col('percentage_2'))
                .then(pl.col('hex'))
                .otherwise(pl.col('hex_2')),
                pl.col('percentage') + pl.col('percentage_2')
            )

            sample_colors_with_new_percentage = (
                sum_of_percentages
                .join(self.colors, how='left', on='hex')
                .drop('percentage_right')
            )

            color_must_remove = pl.concat([with_delta_e['hex'], with_delta_e['hex_2']])

            self.colors = pl.concat([
                sample_colors_with_new_percentage['hex', 'L', 'A', 'B', 'percentage'],
                self.colors.join(color_must_remove.to_frame(), how='anti', on='hex')
            ])

            if self.colors['percentage'].sort()[-self.number_of_most:].sum() > self.sum_of_percentage:
                break

        return self.colors.sort('percentage', descending=True).select('hex', 'percentage').to_dicts()
