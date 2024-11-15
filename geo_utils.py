import ee
import geemap
import wxee
import xarray as xr
from typing import Union, Literal, Dict

def get_study_areas() -> Dict[str, ee.Geometry.Polygon]:
    """Returns dictionary of predefined study areas"""
    areas = {
        'border_area': ee.Geometry.Polygon([
            [[-61.831221, -25.658720],
             [-61.831221, -25.778083],
             [-61.711858, -25.778083],
             [-61.711858, -25.658720]]
        ]),
        'parana_delta': ee.Geometry.Polygon([
            [[-59.5, -33.5],
             [-59.5, -34.0],
             [-58.8, -34.0],
             [-58.8, -33.5]]
        ])
    }
    return areas