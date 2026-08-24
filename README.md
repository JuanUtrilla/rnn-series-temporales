# GRU para predicción de temperatura

Serie diaria de temperatura, ventanas de 5 días de historia y predicción a 2 días vista.
Módulo de *Predicción de series temporales con redes recurrentes* del Máster en Big Data,
Data Science e IA de la Universidad Complutense (curso 2025-26).

El primer paso es el que convierte el problema en supervisado sin romper el orden
cronológico: ventaneado con `create_windows_np`, `past, future = (5, 2)`. A partir de ahí,
dos capas GRU (128 y 64) y una densa de salida.

Ese modelo base se queda en **7,04 de MSE** sobre el test, en °C².

## Añadir contexto al modelo

La segunda parte pide enriquecer la serie con más variables. El enunciado sugería una
`portion_year`; acabaron entrando cinco señales: codificación cíclica seno/coseno del día
del año y del mes, y la media móvil de los últimos 7 días. Las cíclicas le dan la
estacionalidad masticada —diciembre y enero quedan cerca, cosa que un número del 1 al 365
no consigue—, y la media móvil aporta la tendencia reciente. El modelo crece con
*dropout* de 0,2 entre las GRU.

Sobre el objetivo estandarizado, **MSE de 0,365** y **MAE de 1,93 °C** en grados reales.

Un matiz honesto: los dos números de arriba no son comparables entre sí tal cual, porque
el modelo base se evalúa en °C² y el mejorado sobre el objetivo tipificado. Para afirmar
la mejora con rigor habría que devolver ambos a la misma escala y volver a medir. Está sin
hacer.

## Y algo de teoría

El cuaderno cierra razonando dos cosas: por qué esta tarea es *many-to-one* y qué
configuraciones secuencia-a-secuencia tendrían sentido en otros problemas, y el papel de
los *word embeddings* como representación densa frente al *one-hot*.

## Ejecución

Recomendado en Colab por el coste de entrenamiento; en local:

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
jupyter lab rnn_series_temporales.ipynb
```
