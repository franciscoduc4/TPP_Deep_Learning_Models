# Deep Learning Models

## Predicción de Dosis de Insulina Bolus con Modelos de Machine Learning
Este repositorio contiene la implementación de modelos de machine learning para predecir dosis de insulina bolus (normal) en pacientes con diabetes tipo 1, utilizando el dataset DiaTrend con datos de 54 sujetos. El objetivo es asistir a los pacientes en la gestión de su glucosa, mejorando la precisión de las dosis frente a métodos tradicionales basados en reglas.

## Contenido
* Baseline (Modelo Basado en Reglas): Implementa una regla simple para calcular la dosis de insulina usando carbInput, bgInput, insulinCarbRatio e insulinSensitivityFactor.

* Modelo FNN: Una red neuronal feedforward para predecir dosis, usando lecturas de CGM y otras características (carbInput, bgInput, etc.).

* Modelo LSTM: Versión optimizada del LSTM con capas apiladas, BatchNormalization, learning rate ajustado, características adicionales (hour_of_day, insulinSensitivityFactor personalizado) y una función de pérdida personalizada para penalizar sobrepredicciones.

* Modelo TCN: Combina convoluciones causales con redes residuales para procesar secuencias de manera eficiente, capturando patrones locales y globales en las lecturas de CGM con menos riesgo de gradientes que desaparecen, en comparación con los LSTMs.

## Requisitos
Python 3.8+
Librerías: pandas, numpy, scikit-learn, tensorflow, matplotlib, joblib
