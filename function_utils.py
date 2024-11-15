import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional
from pathlib import Path

def compare_dw_maps(
        dataset1: Union[str, Path, xr.Dataset],
        dataset2: Union[str, Path, xr.Dataset],
        data_var1: str = 'label',
        data_var2: str = 'label',
        title_prefix: str = 'Dynamic World Comparison',
        labels1_name: str = 'Map 1',
        labels2_name: str = 'Map 2',
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'Blues',
) -> dict:
    """
    Compara dos mapas de Dynamic World, calculando matriz de confusión y métricas.

    Parameters
    ----------
    dataset1, dataset2 : Union[str, Path, xr.Dataset]
        Datasets de Dynamic World a comparar
    data_var1, data_var2 : str
        Nombres de las variables a comparar en cada dataset
    title_prefix : str
        Prefijo para el título del gráfico
    labels1_name, labels2_name : str
        Nombres para identificar cada mapa
    figsize : Tuple[int, int]
        Tamaño de la figura
    cmap : str
        Colormap para la matriz de confusión

    Returns
    -------
    dict
        Diccionario con matriz de confusión y métricas
    """
    # Diccionario de clases de Dynamic World
    dw_classes = {
        0: 'Water',
        1: 'Trees',
        2: 'Grass',
        3: 'Flooded vegetation',
        4: 'Crops',
        5: 'Shrub & scrub',
        6: 'Built',
        7: 'Bare',
        8: 'Snow & ice'
    }

    def load_dataset(ds):
        if isinstance(ds, (str, Path)):
            return xr.open_dataset(ds)
        return ds

    ds1 = load_dataset(dataset1)
    ds2 = load_dataset(dataset2)

    # Verificar forma
    if ds1[data_var1].shape != ds2[data_var2].shape:
        raise ValueError(f"Los datasets deben tener la misma forma. "
                         f"Se obtuvo {ds1[data_var1].shape} y {ds2[data_var2].shape}")

    # Crear matriz de confusión
    labels = sorted(dw_classes.keys())
    conf_matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            conf_matrix[i, j] = np.sum((ds1[data_var1].values == label1) &
                                       (ds2[data_var2].values == label2))

    # Calcular métricas
    overall_accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

    # Crear etiquetas legibles
    readable_labels = [dw_classes[i] for i in labels]

    # Plotear
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap=cmap,
                xticklabels=readable_labels, yticklabels=readable_labels)
    plt.title(f'{title_prefix}: {labels1_name} vs {labels2_name}')
    plt.xlabel(labels2_name)
    plt.ylabel(labels1_name)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print(f"Precisión general: {overall_accuracy:.2%}")

    return {
        'confusion_matrix': conf_matrix,
        'labels': labels,
        'readable_labels': readable_labels,
        'overall_accuracy': overall_accuracy
    }



import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional, Dict
from pathlib import Path

def compare_dw_esa(
        dw_dataset: Union[str, Path, xr.Dataset],
        esa_dataset: Union[str, Path, xr.Dataset],
        dw_var: str = 'label',
        esa_var: str = 'Map',
        custom_mapping: Dict[int, int] = None,
        title: str = 'Dynamic World vs ESA Land Cover',
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = 'YlOrRd'
) -> dict:
    """
    Compara mapas de Dynamic World y ESA, creando matrices de confusión y calculando métricas.

    Parameters
    ----------
    dw_dataset : Union[str, Path, xr.Dataset]
        Dataset de Dynamic World
    esa_dataset : Union[str, Path, xr.Dataset]
        Dataset de ESA
    dw_var : str
        Nombre de la variable en el dataset de DW
    esa_var : str
        Nombre de la variable en el dataset de ESA
    custom_mapping : Dict[int, int], opcional
        Mapeo personalizado de clases ESA -> DW. Si no se proporciona, se usa el mapeo por defecto
    title : str
        Título para los gráficos
    figsize : Tuple[int, int]
        Tamaño de la figura
    cmap : str
        Colormap para las matrices de confusión

    Returns
    -------
    dict
        Diccionario con matrices de confusión y métricas
    """
    # Definición de clases
    DW_CLASSES = {
        0: 'Water',
        1: 'Trees',
        2: 'Grass',
        3: 'Flooded vegetation',
        4: 'Crops',
        5: 'Shrub & scrub',
        6: 'Built',
        7: 'Bare',
        8: 'Snow & ice'
    }

    ESA_CLASSES = {
        10: 'Tree cover',
        20: 'Shrubland',
        30: 'Grassland',
        40: 'Cropland',
        50: 'Built-up',
        60: 'Bare / sparse vegetation',
        70: 'Snow and ice',
        80: 'Permanent water bodies',
        90: 'Herbaceous wetland',
        95: 'Mangroves',
        100: 'Moss and lichen'
    }

    # Mapeo por defecto ESA -> DW
    DEFAULT_MAPPING = {
        10: 1,    # Tree cover -> Trees
        20: 5,    # Shrubland -> Shrub & scrub
        30: 2,    # Grassland -> Grass
        40: 4,    # Cropland -> Crops
        50: 6,    # Built-up -> Built
        60: 7,    # Bare/sparse -> Bare
        70: 8,    # Snow and ice -> Snow & ice
        80: 0,    # Water bodies -> Water
        90: 3,    # Herbaceous wetland -> Flooded vegetation
        95: 3,    # Mangroves -> Flooded vegetation
        100: 2    # Moss and lichen -> Grass
    }

    # Usar mapeo personalizado si se proporciona
    class_mapping = custom_mapping if custom_mapping is not None else DEFAULT_MAPPING

    def load_dataset(ds):
        if isinstance(ds, (str, Path)):
            return xr.open_dataset(ds)
        return ds

    ds1 = load_dataset(dw_dataset)
    ds2 = load_dataset(esa_dataset)

    # Verificar forma
    if ds1[dw_var].shape != ds2[esa_var].shape:
        raise ValueError(f"Los datasets deben tener la misma forma. "
                         f"Se obtuvo {ds1[dw_var].shape} y {ds2[esa_var].shape}")

    # Matriz de confusión original
    labels_dw = sorted(DW_CLASSES.keys())
    labels_esa = sorted(ESA_CLASSES.keys())

    conf_matrix = np.zeros((len(labels_dw), len(labels_esa)), dtype=int)

    for i, label1 in enumerate(labels_dw):
        for j, label2 in enumerate(labels_esa):
            conf_matrix[i, j] = np.sum((ds1[dw_var].values == label1) &
                                       (ds2[esa_var].values == label2))

    # Plotear matriz original
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap=cmap,
                xticklabels=[ESA_CLASSES[i] for i in labels_esa],
                yticklabels=[DW_CLASSES[i] for i in labels_dw])
    plt.title(f'{title}\nOriginal Confusion Matrix')
    plt.xlabel('ESA Land Cover')
    plt.ylabel('Dynamic World')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Crear array mapeado
    ds2_mapped = ds2[esa_var].values.copy()
    for esa_class, dw_class in class_mapping.items():
        ds2_mapped[ds2[esa_var].values == esa_class] = dw_class

    # Matriz de confusión mapeada
    mapped_labels = sorted(set(DW_CLASSES.keys()))
    conf_matrix_mapped = np.zeros((len(mapped_labels), len(mapped_labels)), dtype=int)

    for i, label1 in enumerate(mapped_labels):
        for j, label2 in enumerate(mapped_labels):
            conf_matrix_mapped[i, j] = np.sum((ds1[dw_var].values == label1) &
                                              (ds2_mapped == label2))

    # Calcular métricas
    overall_accuracy = np.sum(np.diag(conf_matrix_mapped)) / np.sum(conf_matrix_mapped)
    per_class_accuracy = {}
    for i, label in enumerate(mapped_labels):
        class_sum = np.sum(conf_matrix_mapped[i, :])
        if class_sum > 0:
            per_class_accuracy[label] = conf_matrix_mapped[i, i] / class_sum

    # Plotear matriz mapeada
    plt.figure(figsize=figsize)
    mapped_labels_names = [DW_CLASSES[i] for i in mapped_labels]
    sns.heatmap(conf_matrix_mapped, annot=True, fmt='g', cmap=cmap,
                xticklabels=mapped_labels_names,
                yticklabels=mapped_labels_names)
    plt.title(f'{title}\nMapped Classes Confusion Matrix')
    plt.xlabel('ESA (mapped to DW classes)')
    plt.ylabel('Dynamic World')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print(f"\nOverall Accuracy: {overall_accuracy:.2%}")
    print("\nPer-class Accuracy:")
    for label, acc in per_class_accuracy.items():
        print(f"{DW_CLASSES[label]}: {acc:.2%}")

    return {
        'original_confusion_matrix': conf_matrix,
        'mapped_confusion_matrix': conf_matrix_mapped,
        'overall_accuracy': overall_accuracy,
        'per_class_accuracy': per_class_accuracy,
        'dw_classes': DW_CLASSES,
        'esa_classes': ESA_CLASSES,
        'class_mapping_used': class_mapping
    }


import ee
import geemap
import wxee
import xarray as xr
from typing import Union, Literal, Dict

def initialize_gee(project_id: str, opt_url: str = 'https://earthengine-highvolume.googleapis.com') -> None:
    """Initialize Google Earth Engine with authentication"""
    ee.Authenticate()
    ee.Initialize(project=project_id, opt_url=opt_url)

def process_dynamic_world(
        aoi: ee.Geometry,
        processing_type: Literal['mode', 'mode_pooled', 'mean_pooled', 'mean_unpooled'],
        output_path: str,
        scale: int = 30,
        start_date: str = "2020-01-01",
        end_date: str = "2021-01-01"
) -> None:
    """
    Process Dynamic World imagery with specified parameters

    Args:
        aoi: Area of interest geometry
        processing_type: Type of processing to apply
        output_path: Where to save the output NetCDF
        scale: Resolution in meters
        start_date: Start date for imagery
        end_date: End date for imagery
    """
    CLASS_NAMES = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops',
                   'shrub_and_scrub', 'built', 'bare', 'snow_and_ice']
    CLASS_NUMBER = ee.List.sequence(0, 8)

    dw_filtered = wxee.TimeSeries("GOOGLE/DYNAMICWORLD/V1") \
        .filterDate(start_date, end_date) \
        .filterBounds(aoi) \
        .map(lambda img: img.clip(aoi))

    if processing_type == 'mode':
        ts = dw_filtered.select("label").aggregate_time("year", ee.Reducer.mode())

    elif processing_type in ['mode_pooled', 'mean_pooled']:
        reducer = ee.Reducer.mode() if processing_type == 'mode_pooled' else ee.Reducer.mean()
        ts = dw_filtered.select("label" if processing_type == 'mode_pooled' else CLASS_NAMES) \
            .aggregate_time("year", reducer) \
            .map(lambda img: img.setDefaultProjection('EPSG:4326', None, scale) \
                 .reduceResolution(
            reducer=reducer,
            maxPixels=65535,
            bestEffort=True
        ).reproject(crs='EPSG:4326', scale=scale))

    else:  # mean_unpooled
        ts = dw_filtered.aggregate_time("year", ee.Reducer.mean())
        ts = ts.map(
            lambda image:
            image.select(CLASS_NAMES)
            .reduce(ee.Reducer.max())
            .eq(image.select(CLASS_NAMES))
            .multiply(ee.Image.constant(CLASS_NUMBER))
            .reduce(ee.Reducer.sum())
            .rename('label')
            .uint8()
            .copyProperties(image, ['system:time_start'])
        )

    ds = ts.wx.to_xarray(region=aoi, scale=scale)
    ds = ds.fillna(-1).clip(min=-128, max=127)
    ds["label"] = ds["label"].astype("int8")
    ds["x"] = ds["x"].astype("float32")
    ds["y"] = ds["y"].astype("float32")
    ds["time"] = ds["time"].dt.strftime("%Y-%m")
    ds.to_netcdf(output_path)