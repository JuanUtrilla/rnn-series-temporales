# Predicción de temperatura con redes recurrentes

¿Se puede predecir la temperatura mínima de pasado mañana a partir de la de los últimos
cinco días? Con diez años de mínimas diarias de Melbourne, una red neuronal que lee los
datos en orden acierta con un error medio de unos 2 °C si también sabe en qué época del
año está.

Ejercicio del módulo de Predicción de series temporales con redes recurrentes del Máster
en Big Data, Data Science e IA de la Universidad Complutense (curso 2025-26).

## Cómo está hecho

Los 3.000 primeros días sirven para entrenar y los 650 últimos para probar. La serie se
corta en ventanas de 5 días, cada una emparejada con la temperatura 2 días después.

El modelo base (dos capas GRU de 128 y 64 unidades) tiene un error cuadrático medio de
7,04 °C² en prueba. La segunda versión añade el día del año y el mes en seno y coseno,
para que diciembre y enero queden cerca, y la media de los últimos 7 días. Su error
absoluto medio en prueba es 1,93 °C.

Ojo: las dos cifras no se pueden comparar tal cual, porque usan métricas y escalas
distintas. Medir los dos modelos igual para confirmar la mejora está pendiente.

El cuaderno termina con dos preguntas de teoría sobre arquitecturas *many-to-one* y
*word embeddings*.

## Ejecución

Mejor en Colab por el coste de entrenamiento. En local:

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
jupyter lab rnn_series_temporales.ipynb
```
