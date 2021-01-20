import dask


def blue_func(src):
    blue = src['B02']
    return blue


def swir2_func(src):
    swir2 = src['B12']
    return swir2