# Registro de Decisiones del Proyecto



## Fecha: [2024-27-11]
### Tarea: Creo esquema de muestreo estratificado en `Mustreo_estratificado.ipynb`
- **Estado de la tarea:**
- [x] Establecer ecoregiones para la estratificación
- [ ] Cruzar las ecoregiones con un índice espacial (H3 o s2)
- [ ] Decidir si vamos a utilizar H3 o S2 como índice espacial
- [ ] Decidir si vamos a hacer una diferenciación entre áreas puras y áreas fronterizas donde se encuentra más de una ecoregión.
- [ ] Decidir si vamos a actualizar nuestra área de estudio. Hasta ahora era Argentina menos Patagonia. Podríamos decidir un criterio de ecoregiones en vez de político.

- **Contexto:**
En base a charlas con Germán decidimos que era necesario establecer un sistema de muestreo estratificado para validar DW de forma representativa sobre todo el territorio.
La idea es estratificar por ecoregiones.

- **Decisiones:**
  - Utilizar la base de datos de [WWF](https://www.worldwildlife.org/publications/terrestrial-ecoregions-of-the-world) para las ecoregiones. [Paper correspondiente.](https://academic.oup.com/bioscience/article/51/11/933/227116)

- **Alternativas:**
  1. No busqué alternativas a WWF

- **Razonamiento:**

- **Implicaciones:**

- **Notas de Germán:**

- **Resultados:**


## Fecha: [2024-28-11]
### Tarea 1: Descarga de imágenes de casos de estudio específicos para establecer un workflow de validación.

- **Estado de la tarea:** En proceso


Descargo imágenes de ESA y DW a 30m y 50m metros de resolución sobre dos áreas de interés: Chaco y el Delta del Paraná. Las geometrías están en `geo_utils.py`
La idea es compararlas y tomar una decisión sobre el formato de descarga y el proceso de validación.
Una vez establecido el proceso de validación la idea es aplicarlo de forma representativa sobre un muestreo estratificado de nuestra área total de estudio _a definir_.
- **Contexto:**
En base a conversaciones con Germán queremos establecer una metodología para poder validar los resultados de DW,
así que elegimos dos casos de estudio para poder comparar en un intento de "verdad de campo" contra ESA WorldCover.
Adémas estamos validando los métodos de descarga de DW, ya que hay distintas formas de agregar los datos tento temporal como espacialmente.
- **Decisiones:** 
  - Estoy descargando imágenes solo a 30m y 50m:
  - Utilizo 4 métodos de descarga distintos (moda, moda pooleada, media y media pooleada)
- **Alternativas:**
    1. Descargar también a escala de 10m.
    2. Utilizar otras áreas de estudio
    3. Descargar también las imágenes Sentinel-2 crudas a partir de las cuales DW clasifica y calcular índices espectrales.
- **Razonamiento:**

- **Implicaciones:** 

- **Notas de Germán:**

- **Resultados:**


# Template

## Fecha: [2024-28-11]
### Tarea x: 

- **Contexto:** 
  - 
 
- **Decisiones:**
    -
  
- **Alternativas:**
    1.
    2. 
    3. 
- **Razonamiento:**

- **Implicaciones:**