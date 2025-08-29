import polars as pl


class Lab:
    @staticmethod
    def __rgb_to_xyz(rgb: pl.DataFrame, /) -> pl.DataFrame:
        return (
            rgb
            .select(pl.all() / 255)
            .select(
                pl
                .when(pl.all() <= 0.04045).then(pl.all() / 12.92)
                .otherwise(((pl.all() + 0.055) / 1.055) ** 2.4)
            )
            .select(
                x=pl.col('r') * 0.4124564 + pl.col('g') * 0.3575761 + pl.col('b') * 0.1804375,
                y=pl.col('r') * 0.2126729 + pl.col('g') * 0.7151522 + pl.col('b') * 0.0721750,
                z=pl.col('r') * 0.0193339 + pl.col('g') * 0.1191920 + pl.col('b') * 0.9503041
            )
        )

    @staticmethod
    def __xyz_to_lab(_xyz: pl.DataFrame, /) -> pl.DataFrame:
        return (
            _xyz.select(
                xr=pl.col('x') / 95.047,
                yr=pl.col('y') / 100.000,
                zr=pl.col('z') / 108.883
            )
            .select(
                pl
                .when(pl.all() > 0.008856)
                .then(pl.all() ** (1 / 3))
                .otherwise((7.787 * pl.all()) + (16 / 116))
            )
            .select(
                L=(116 * pl.col('xr')) - 16,
                A=500 * (pl.col('xr') - pl.col('yr')),
                B=200 * (pl.col('yr') - pl.col('zr'))
            )
        )

    @staticmethod
    def rgb_to_lab(rgb: pl.DataFrame, /) -> pl.DataFrame:
        xyz = Lab.__rgb_to_xyz(rgb)
        lab = Lab.__xyz_to_lab(xyz)
        return Lab.__with_hex(pl.concat([rgb, lab], how='horizontal'))

    @staticmethod
    def __with_hex(image: pl.DataFrame, /) -> pl.DataFrame:
        def func(colors: dict):
            def to_hex(number: int):
                return hex(number)[2:].upper()

            return '#' + to_hex(colors['r']) + to_hex(colors['g']) + to_hex(colors['b'])

        return (
            image
            .select(
                pl.struct('r', 'g', 'b')
                .map_elements(func, return_dtype=pl.Utf8)
                .alias('hex'),
                'L', 'A', 'B'
            )
        )
