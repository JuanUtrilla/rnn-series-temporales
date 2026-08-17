# Redes recurrentes para predicción de series temporales

Predicción de temperatura a varios días vista con **redes recurrentes (GRU)** en
TensorFlow/Keras, sobre un dataset de series temporales meteorológicas.

Desarrollado en el módulo de *Predicción de series temporales con redes recurrentes* del
Máster en Big Data, Data Science e Inteligencia Artificial (Universidad Complutense de
Madrid, curso 2025-26).

## Contenido

- **Ventaneado de la serie** — transformación de la serie continua en ventanas
  deslizantes de tamaño 5 para predecir el valor a 2 días, que es el paso que convierte un
  problema temporal en uno supervisado sin romper el orden cronológico.
- **Modelo recurrente** — arquitectura de dos capas **GRU** sobre las ventanas generadas.
- **Enriquecimiento con features** — incorporación de variables adicionales, entre ellas
  `portion_year` (posición dentro del año), que da al modelo la estacionalidad de forma
  explícita en lugar de obligarlo a inferirla; el modelo resultante mejora al anterior.
- **Fundamentos** — razonamiento sobre arquitecturas *many-to-one* frente a otras
  configuraciones secuencia-a-secuencia, y sobre el papel de los *word embeddings* como
  alternativa densa al *one-hot encoding*.

## Ejecución

Recomendado en **Google Colab** por el coste de entrenamiento; también funciona en local:

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
jupyter lab rnn_series_temporales.ipynb
```

## Stack

TensorFlow · Keras · NumPy · pandas · Matplotlib

---

**Juan Peñas Utrilla** — [LinkedIn](https://www.linkedin.com/in/juan-penas-utrilla/) · [GitHub](https://github.com/JuanUtrilla)
