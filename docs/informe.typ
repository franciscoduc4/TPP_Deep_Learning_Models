// Imports
#import "@preview/fletcher:0.5.4" as fletcher: diagram, node, edge
#import "@preview/showybox:2.0.3": *
#import "@preview/colorful-boxes:1.4.2": *
#import "@preview/chronos:0.2.0"
#import "@preview/thmbox:0.1.1": *

// Texto
#set par(justify: true)
#set text(lang: "es")

// Tipografías
#show raw: set text(font: "Fira Code")
// #show math.equation: set text(font: "Fira Math")
#show sym.emptyset: set text(font: "Fira Sans")
#show ref: set text(teal.darken(33%))
#show link: set text(blue.darken(13%))
#show ref: it => {
  if it.element != none and it.element.func() == heading {
    let l = it.target // label
    let h = it.element // heading
    link(l, [#h.body [p. #h.location().page()]])
  } else {
    it
  }
}

// Configuración de títulos, ítems y listas
#set heading(numbering: "1.")
#set enum(full: true)

// Ecuaciones
// #show math.equation: set align(center)
// #set math.equation(numbering: n => {
//   numbering("(1.1)", counter(heading).get().first(), n)
// })

// Recuadros
#let recuadro(color, titulo, cuerpo, pie: "") = [
  #if pie.trim() == "" {
    showybox(
      frame: (
        border-color: color.darken(50%),
        title-color: color.lighten(60%),
        body-color: color.lighten(80%)
      ),
      title-style: (
        color: black,
        weight: "regular",
        align: center
      ),
      shadow: (
        offset: 3pt,
      ),
      title: [*#titulo*],
      cuerpo
    )
  } else {
      showybox(
      frame: (
        border-color: color.darken(50%),
        title-color: color.lighten(60%),
        body-color: color.lighten(80%)
      ),
      title-style: (
        color: black,
        weight: "regular",
        align: center
      ),
      shadow: (
        offset: 3pt,
      ),
      title: [*#titulo*],
      cuerpo,
      [#text(pie, style: "italic")]
    )
  }
]

#let definicion(concepto, definicion, colour: "blue") = [
  #colorbox(
    title: concepto,
    color: colour,
    radius: 2pt,
    width: auto
  )[
    #definicion
  ]
]

#let pregunta(pregunta, respuesta) = [
  #outline-colorbox(
    title: text(size: 14pt)[*Pregunta de Examen*],
    color: "teal",
    width: auto,
    radius: 7pt,
    centering: true
  )[
    *#pregunta*

    #respuesta
  ]
]

#let bordear(titulo, codigo, colour: "gold") = [
  #outline-colorbox(
    title: text(size: 14pt)[#titulo],
    color: colour,
    width: auto,
    radius: 7pt,
    centering: true
  )[
    #codigo
  ]
]

#let resaltar(
  color: rgb("#118d8d"),
  fill: rgb("#118d8d").lighten(83%),
  variant: "Título",
  thmbox: thmbox,
  ..args
) = note(fill: fill, variant: variant, color: color, ..args) 


#let TODO(body) = bordear("TODO", body, colour: "red")


// Referencias
#let pageref(label) = context {
  let loc = locate(label)
  let nums = counter(page).at(loc)
  link(loc, numbering(loc.page-numbering(), ..nums))
}

// Celdas
#let ventaja(text) = table.cell(
  text,
  align: center,
  fill: lime.lighten(53%)
)

#let desventaja(text) = table.cell(
  text,
  align: center,
  fill: red.lighten(53%)
)

// Título y subtítulos
#let title(titulo) = [#align(center, text(titulo, size: 20pt, weight: "extrabold"))]
#let subtitulo(subtitulo) = [#align(center, text(silver.darken(70%), size: 16pt)[#subtitulo])]
#let encabezado(encabezado) = [#align(left, text(maroon, size: 14pt)[#encabezado])]

#title("Modelos de Aprendizaje Profundo")

#outline(title: "Índice", depth: 5)<index>

#pagebreak()
#counter(page).update(1)

// Configuración de la hoja
#show heading.where(level: 1): it => pagebreak(weak: true) + it
#set heading(numbering: "1.")
#set page(
  paper: "a4",
  margin: 2cm,
  numbering: "1",
  header: context {
    set align(center)
    set text(size: 8pt)
    [#text(size: 10pt)[Modelos de Aprendizaje Profundo]]
    line(length: 100%, stroke: .25pt)
  },
  footer: context [
    #line(length: 100%, stroke: .25pt)
    #let headings = query(
      selector(heading.where(level: 1)).before(here())
    )
    
    #if headings == () { return }
    
    #let current = headings.last()
    
    #set text(size: 10pt)
    #let number = counter(heading).at(current.location())
    #number.at(0). #current.body | #link(<index>, "Índice")
    #h(1fr)
    Página #counter(page).display(
      both: true,
      "1 de 1"
    )
  ]
)

= Resumen <abstract>

El presente informe tiene como objetivo presentar la documentación de los modelos de aprendizaje profundo utilizados en el trabajo práctico. Se incluye una comparación entre los modelos implementados, FNN (Feedforward Neural Network), TCN (Temporal Convolutional Network), GRU (Gated Recurrent Unit), PPO (Proximal Policy Optimization), entre otros modelos, donde se analizan las distintas métricas obtenidas, como el MAE (Mean Absolute Error), RMSE (Root Mean Square Error), R#super[2] (R-squared), además de una comparativa entre el uso de CPU y GPU.

= Introducción <introduccion>

El presente documento reúne la documentación de los modelos de aprendizaje profundo utilizados en el trabajo práctico, así también como una breve descripción de la implementación, utilidad y resultados obtenidos de cada uno. Se incluye una comparativa entre los modelos para poder obtener una conclusión sobre cuál es el modelo más adecuado a utilizar.

= Objetivos <objetivos>

El objetivo de esta sección es establecer cuáles de los modelos son adecuados para usar en el trabajo práctico, comparando las distintas métricas obtenidas y su eficiencia.

Las métricas a comparar son las siguientes:

1. *MAE* (_Mean Absolute Error_, Error Absoluto Medio) @mae: mide la magnitud promedio de los errores en un conjunto de predicciones, sin considerar su dirección. Se calcula como la media de las diferencias absolutas entre los valores predichos y los valores reales.

#resaltar(
  variant: "Fórmula MAE",
  [
    #align(center)[$"MAE" = 1/n * sum abs(y_i - hat(y)_i)$]

    donde:
    - $n$ es el número de predicciones.
    - $y_i$ es el valor real.
    - $hat(y)_i$ es el valor predicho.
  ]
)

Se busca #highlight[#underline[*minimizar*]] el MAE, ya que un MAE bajo indica que el modelo tiene un buen rendimiento en la predicción de los valores, lo que significaría, en el marco de este trabajo, una predicción precisa de dosis de insulina. Idealmente, se busca obtener un valor por debajo de 0.5 unidades.

2. *RMSE* (_Root Mean Square Error_, Raíz del Error Cuadrático Medio) @rmse: mide el desvío estándar de los errores de predicción. A diferencia del MAE, el RMSE penaliza los errores más grandes de maenra más significativa dado el término cuadrático, por lo que un RMSE más bajo indica que el modelo tiene errores más pequeños en general y es más consistente en sus predicciones. Se calcula como la raíz cuadrada del promedio de los cuadrados de las diferencias entre los valores predichos y los valores reales.

#resaltar(
  variant: "Fórmula RMSE",
  [
    #align(center)[$"RMSE" = sqrt(1/n * sum (y_i - hat(y)_i)^2)$]

    donde:
    - $n$ es el número de predicciones.
    - $y_i$ es el valor real.
    - $hat(y)_i$ es el valor predicho.
  ]
)

Se busca #highlight[#underline[*minimizar*]] el RMSE, ya que un RMSE bajo indica que el modelo tiene un buen rendimiento en la predicción de los valores, lo que significaría, en el marco de este trabajo, una predicción precisa de dosis de insulina. Idealmente, se busca obtener un valor por debajo de 1.0 unidad.

3. *R#super[2]* (_R-squared_, Coeficiente de Determinación) @rsquared: mide la proporción de la varianza total en los datos que es explicada por el modelo. Un R#super[2] más alto indica que el modelo explica mejor la variabilidad de los datos, lo que significa que tiene un mejor ajuste. Se calcula como:

#resaltar(
  variant: [Fórmula R#super[2]],
  [
    #align(center)[$R^2 = 1 - "SS"_"RES" / "SS"_"TOT" = 1 - (sum (y_i - hat(y)_i)^2) / (sum (y_i - macron(y))^2)$]

    donde:
    - SS#sub[RES] es la suma de los cuadrados de los residuos.
    - SS#sub[TOT] es la suma total de los cuadrados.
    - $y_i$ es el valor real.
    - $hat(y)_i$ es el valor predicho.
    - $macron(y)$ es la media de los valores reales.
  ]
)

Un R#super[2] de 1 indica que el modelo explica toda la variabilidad de los datos, mientras que un R#super[2] de 0 indica que el modelo no explica nada de la variabilidad. Se busca #highlight[#underline[*maximizar*]] el R#super[2], idealmente por encima de 0.9, que indicaría que el modelo explica más del 90% de la varianza en los datos y reflejaría un ajuste excelente.

4. *Estabilidad por Sujeto*: se busca lograr que las métricas sean consistentes entre sujetos con especial atención en la reducción del MAE y mejorar el R#super[2] en casos problemáticos.

= Modelos <modelos>

== Modelos de Aprendizaje Profundo <deep-learning>

=== Baseline (Modelo Basado en Reglas) <baseline>
==== Descripción <descripcion-baseline>

El modelo Baseline es un enfoque tradicional que usa un conjunto de reglas predefinidas por expertos para calcular la dosis de insulina. En este modelo, se determina la dosis mediante la siguiente fórmula:

#resaltar(
  color: rgb("#a1c22d"),
  fill: rgb("#a1c22d").lighten(83%),
  variant: "Fórmula Baseline",
  [
    #align(center)[
      $"dp" = "ci"/"icr" + ("bgi" - "tbg") / "isf"$
    ]

    donde:
    - $"dp"$ es la dosis predicha.
    - $"ci"$ (_carbInput_) es la cantidad de carbohidratos de entrada.
    - $"icr"$ (_insulinCarbRatio_) es la relación de carbohidratos a insulina.
    - $"bgi"$ (_bgInput_) es la cantidad de glucosa de entrada.
    - $"tbg"$ (_targetBg_) es la glucosa objetivo.
    - $"isf"$ (_insulinSensitivityFactor_) es el factor de sensibilidad a la insulina.
  ]
)

==== Componentes Principales <componentes-baseline>

+ #highlight(fill: lime.lighten(43%))[*Entradas*]
  - *ci* (_carbInput_): cantidad de carbohidratos de entrada.
  - *bgInput* (_bgInput_): cantidad de glucosa de entrada.
  - *icr* (_insulinCarbRatio_): relación de carbohidratos a insulina.
  - *isf* (_insulinSensitivityFactor_): factor de sensibilidad a la insulina.
+ #highlight(fill: lime.lighten(43%))[*Regla de Cálculo*]
  - La dosis de insulina se calcula sumando la insulina necesaria para cubrir los carbohidratos y la insulina necesaria para corregir el nivel de glucosa actual al objetivo.

==== Ventajas en la Predicción de Glucosa <ventajas-baseline>
#list(
  marker: ([✅], [✓]),
  [De fácil entendimiento e implementación.],
  [No reuqiere de datos históricos extensos para su funcionamiento inicial.],
  [Puede servir como punto de referencia para comparar el rendimiento de los modelos más complejos.]
)

==== Consideraciones Importantes <consideraciones-baseline>

#list(
  marker: [⚠︎],
  [La precisión depende en gran medida de la correcta configuración de las reglas y los parámetros individuales del paciente.],
  [Puede no adaptarse bien a la variabilidad individual y a patrones complejos en los datos de glucosa.],
  [No aprende de los datos ni mejora con el tiempo.]
)

=== FNN (Feedforward Neural Network) <fnn>
==== Descripción <descripcion-fnn>

Feedforward Neural Network (FNN, Red Neuronal Feedforward) es un tipo de red neuronal artificial donde las conexiones entre los nodos (las neuronas) no forman un ciclo.La información se mueve en una sola dirección, desde la capa de entrada, a través de las capas ocultas (si las hay), hasta la capa de salida. En este contexto, el FNN se utiliza para predecir la dosis de insulina basándose en las lecturas del monitor continuo de glucosa (CGM) y otras características relevantes en un momento dado.

==== Componentes Principales <componentes-fnn>

+ #highlight(fill: lime.lighten(43%))[*Capa de Entrada*]
  - Recibe las características relevantes para la predicción, como las lecturas de CGM recientes, ingesta de carbohidratos, nivel actual de glucosa, etc.
+ #highlight(fill: lime.lighten(43%))[*Capas Ocultas*]
  - Realizan transformaciones no lineales en los datos de entrada para aprender patrones complejos.
  - El número de capas ocultas y el número de neuronas en cada capa son hiperparámetros que se ajustan durante el entrenamiento.
+ #highlight(fill: lime.lighten(43%))[*Capa de Salida*]
  - Produce la predicción de la dosis de insulina.
  - En el caso particular de este modelo, la capa de salida tendrá una única neurona con una función de activación adecuada para la predicción de un valor continuo (dosis de insulina).
+ #highlight(fill: lime.lighten(43%))[*Función de Activación*]
  - Introducen la no linealidad a la red, permitiendo aprender relaciones complejas entre las características de entrada y la dosis de insulina.
  - Se utiliza una función de activación como ReLU (Rectified Linear Unit) o Sigmoid en las capas ocultas para introducir no linealidades en el modelo.
  - En la capa de salida, se puede utilizar una función de activación lineal o ninguna función de activación para permitir que la red produzca valores continuos.
+ #highlight(fill: lime.lighten(43%))[*Función de Pérdida*]
  - Se utiliza una función de pérdida como el error cuadrático medio (MSE) para medir la discrepancia entre las predicciones de la red y los valores reales de dosis de insulina.
  - El objetivo del entrenamiento es minimizar esta función de pérdida ajustando los pesos de la red.

==== Ventajas en la Predicción de Glucosa <ventajas-fnn>

#list(
  marker: ([✅], [✓]),
  [Puede aprender relaciones no lineales complejas entre las características y la dosis de insulina.],
  [Es relativamente sencillo de implementar y entrenar.],
  [Puede utilizar diversas características como entrada para mejorar la precisión de la predicción.]
)

==== Consideraciones Importantes <consideraciones-fnn>

#list(
  marker: [⚠︎],
  [No tiene memoria inherente de secuencias temporales, por lo que puede no capturar dependencias a largo plazo en los datos de glucosa.],
  [El rendimiento depende de la calidad y la cantidad de los datos de entrenamiento.],
  [La elección de la arquitectura (número de capas y neuronas) y los hiperparámetros requiere experimntación y ajuste.]
)

=== TCN (Temporal Convolutional Network) <tcn>
==== Descripción <descripcion-tcn>

Temporal Convolutional Network (TCN, Red Convolucional Temporal) es una arquitectura de red neuronal diseñada específicamente para procesar datos secuenciales. A diferencia de las redes neuronales recurrentes (RNN), las TCNs usan convoluciones casuales, lo que significa que la predicción en un momento dado solo depende de los datos pasados y presentes, evitando la 'mirada hacia el futuro'. Además, las TCNs a menudo incorporan redes residuales para facilitar el entrenamiento de redes profundas y mitigar el problema del gradiente desvaneciente @vanishing-gradient.

==== Componentes Principales <componentes-tcn>

+ #highlight(fill: lime.lighten(43%))[*Convoluciones Causales*]
  - Aseguran que la salida en el tiempo $t$ solo dependa de las entradas hasta el tiempo $t$.
  - Se implementan típicamente utilizando convoluciones unidimensionales con un desplazamiento adecuado.
+ #highlight(fill: lime.lighten(43%))[*Redes Residuales*]
  - Permiten que la información fliua directamente a través de las capas, facilitando el aprendizaje de identidades y mejorando el flujo de gradientes.
  - Un bloque residual típico consiste en una o más capas convolucionales seguidas de una conexión de salto que suma la entrada del bloque a su salida.
+ #highlight(fill: lime.lighten(43%))[*Dilatación*]
  - Las convoluciones dilatadas permiten que la red tenga un campo receptivo muy grande con relativamente pocas capas.
  - El factor de dilatación aumenta exponencialmente con la profundidad de la red, permitiendo capturar dependencias a largo plazo en la secuencia.

==== Ventajas en la Predicción de Glucosa <ventajas-tcn>

#list(
  marker: ([✅], [✓]),
  [Procesa secuencias de manera eficiente y en paralelo, pudiendo ser más rápido que las RNNs.],
  [Tiene un campo receptivo flexible que puede adaptarse a la longitud de las dependencias temporales en los datos de glucosa.],
  [Es menos susceptible al porblema de gradiente desvaneciente o explotan en comparación con las RNNs],
  [Puede capturar patrones tanto locales como globales en las series temporales.]
)

==== Consideraciones Importantes <consideraciones-tcn>

#list(
  marker: [⚠︎],
  [Puede requerir más menoria que las RNNs para campos receptivos muy grandes.],
  [La interpretación de los patrones aprendidos puede ser más compleja que en las RNNs.],
  [El diseño de la arquitectura (número de filtros, capas, tasas de dilatación) puede requerir ajustes.]
)

=== GRU (Gated Recurrent Unit) <gru>
==== Descripción <descripcion-gru>

Gated Recurrent Unit (GRU, Unidad Recurrente Cerrada) es un tipo de red neuronal recurrente (RNN) que, al igual que el LSTM (Long Short Term Memory), está diseñada para manejar datos secuenciales y dependencias a largo plazo. Sin embargo, la GRU tiene una arquitectura más sumple con solo dos puertas: una de actualización; otra de reinicio.

La puerta de actualización controla cuánto del estado anterior debe conservarse, y cuánta nueva información debe agregarse. La puerta de reinicio determina cuánto del estado anterior debe olvidarse. Esta simplificación hace que las GRUs sean a menudo más rápidas de entrenar y tengan menos parámetros que las LSTMs, al tiempo que mantienen una capacidad similar para capturar dependencias temporales.

==== Componentes Principales <componentes-gru>

+ #highlight(fill: lime.lighten(43%))[*Puerta de Actualización*]
  - Controla cuánto del estado oculto anterior se mantiene en el estado oculto actual.
  - Ayuda a la red a decidir qué información del pasado debe conservarse para el futuro.
+ #highlight(fill: lime.lighten(43%))[*Puerta de Reinicio*]
  - Determina cuánto del estado oculto anterior se utiliza para calcular el nuevo estado candidato.
  - Ayuda a la red a olvidar información irrelevante del pasado.
+ #highlight(fill: lime.lighten(43%))[*Estado Candidato*]
  - Representa la nueva información que se va a agregar al estado oculto actual.
  - Se calcula utilizando la puerta de reinicio y el estado oculto anterior.
+ #highlight(fill: lime.lighten(43%))[*Estado Oculto*]
  - Almacena la información aprendida de la secuencia hasta el momento.
  - Se actualiza en cada paso de tiempo utilizando las puertas de actualización y reinicio.

==== Ventajas en la Predicción de Glucosa <ventajas-gru>

#list(
  marker: ([✅], [✓]),
  [Captura dependencias temporales en los datos de glucosa.],
  [Maneja secuencias de longitud variable.],
  [Tiene menos parámetros y es más eficiente computacionalmente que el LSTM.],
  [Puede lograr un rendimiento similar al LSTM en muchas tareas de modelado de secuencias.]
)

==== Consideraciones Importantes <consideraciones-gru>

#list(
  marker: [⚠︎],
  [Puede que no capture dependencias a muy largo plazo tan bien como el LSTM en algunos casos.],
  [Al igual que el LSTM, requiere suficientes datos de entrenamiento y es sensible a la escala de los datos.],
  [La longitud de la secuencia y el número de unidades GRU afectan el rendimiento.]
)

== Modelos de Aprendizaje por Refuerzo <reinforcement-learning>


== Modelos de Aprendizaje por Refuerzo Profundo <drl>

=== PPO (Proximal Policy Optimization) <ppo>
==== Descripción <descripcion-ppo>

Proximal Policy Optimization (PPO, Optimización de Políticas Proximal) es un algoritmo de aprendizaje por refuerzo que se usa para entrenar agentes que toman decisiones secuenciales. En el contexto de la predicción de dosis de insulina, el agente (modelo PPO) aprende una política, que es una función que mappea el estado actual del paciente (por ejemplo, lecturas de CGM, ingesta de carbohidratos, actividad física) a una acción (la dosis de insulina a administrar).

El objetivo del agente es aprender una política que maximice una recompensa acumulada a lo largo del tiempo donde la recompensa está diseñada para reflejar el mantenimiento de los niveles de glucosa dentro de un rango saludable.

PPO es un algoritmo _on-policy_, lo que significa que aprende de las experiencias generadas por la política actual y actualiza la política de manera tal que los nuevos comportamientos no se desvíen demasiado de los antiguos, ayudando a estabilizar el entrenamiento.

==== Componentes Principales <componentes-ppo>

+ #highlight(fill: lime.lighten(43%))[*Agente*]
  - El modelo que aprende a tomar decisiones sobre la dosis de insulina a administrar en función del estado actual del paciente.
  - Aprende a través de la interacción con el entorno (paciente) y la retroalimentación recibida.
+ #highlight(fill: lime.lighten(43%))[*Entorno*]
  - La simulación del paciente y su respuesta a las dosis de insulina en función de sus datos (CGM, comidas, etc.)
+ #highlight(fill: lime.lighten(43%))[*Política*]
  - La función que el agente aprende para mappear el estado del entorno a las acciones (dosis de insulina).
  - En PPO, la política suele estar representada por una red neuronal.
+ #highlight(fill: lime.lighten(43%))[*Función de Valor*]
  - Estima la recompensa futura esperada para un estado dado.
  - Se usa para reducir la varianza en las estimaciones de la ventaja.
+ #highlight(fill: lime.lighten(43%))[*Función de Recompensa*]
  - Define el objetivo del agente. En este caso, puede ser una función que otorga recompensas por mantener los niveles de glucosa dentro de un rango objetivo y penaliza las desviaciones.
+ #highlight(fill: lime.lighten(43%))[*Optimización Proximal*]
  - El mecanismo clave de PPO que limita la magnitud del cambio en la política durante cada actualización para evitar grandes caídas en el rendimiento.
  - Utiliza una función objetivo recortada para asegurar que la nueva política no sea demasiado diferente de la política anterior.

==== Ventajas en la Predicción de Glucosa <ventajas-ppo>

#list(
  marker: ([✅], [✓]),
  [Puede aprender políticas óptimas para la administración de insulina a largo plazo, considerando las consecuencias futuras de las decisiones actuales.],
  [Se adapta a la dinámica compleja y a la variabilidad individual de los pacientes.],
  [Puede incorporar múltiples factores y objetivos en la función de recompensa. Por ejemplo, mantener la glucosa en rango, minimizar la hipoglucemia, hiperglucemia, etc.]
)

==== Consideraciones Importantes <consideraciones-ppo>

#list(
  marker: [⚠︎],
  [El entrenamiento de modelos de aprendizaje por refuerzo puede ser complejo y requerir una gran cantidad de datos y simulación del entorno.],
  [La definición de la función de recompensa es crucial y puede afectar el comportamineto del agente.],
  [La interpretabilidad de la política aprendida puede ser un desafío.],
  [La estabilidad del entrenamiento puede ser un problema, y se requieren técnicas como la optimización proximal para mejorarla.]
)

= Herramientas Utilizadas <herramientas>

== Lenguajes de Programación <lenguajes>

=== Python <python>

#recuadro(
  rgb("#FFD348"),
  "Python",
  [
    Se eligió Python como lenguaje de programación para el desarrollo del trabajo práctico debido al gran ecosistema que ofrece para el procesamiento y análisis de datos, así como para el aprendizaje automático y el aprendizaje por refuerzo. Python es ampliamente utilizado en la comunidad científica y tiene una gran cantidad de bibliotecas y herramientas que facilitan la implementación de modelos complejos.

    Entre las librerías que se usaron se encuentran NumPy, Pandas, TensorFlow, Keras y PyTorch. Estas librerías proporcionan funcionalidades avanzadas para el manejo de datos, la construcción de modelos y la optimización de hiperparámetros, que se detallarán más adelante.
  ],
  pie: "Se debe usar Python 3.8 o superior."
)

=== Julia <julia>

#recuadro(
  rgb("#955EA6"),
  "Julia",
  [
    Julia es un lenguaje de programación de alto rendimiento y fácil de usar, especialmente diseñado para la computación técnica y científica. Se eligió Julia para el desarrollo del trabajo práctico debido a su capacidad para manejar cálculos numéricos intensivos y su sintaxis clara y concisa.

    Julia es particularmente adecuada para tareas de aprendizaje automático y aprendizaje por refuerzo, ya que ofrece un rendimiento comparable al de lenguajes como C, al tener un compilador JIT (Just in Time). Además, Julia ofrece varios paquetes para realizar análisis de datos y aprendizaje automático, como Flux.jl y MLJ.jl, que permiten construir y entrenar modelos de manera eficiente.
  ]
)

== Librerías <librerias>

=== Procesamiento y Análisis de los Datos <procesamiento>

==== NumPy <numpy>

#recuadro(
  rgb("#4b73c9"),
  "NumPy",
  [
    NumPy es una biblioteca fundamental para la computación científica en Python. Proporciona un objeto de matriz multidimensional y funciones para trabajar con estos arrays de manera eficiente. NumPy es ampliamente utilizado para realizar operaciones matemáticas y lógicas sobre arrays, así como para manipular datos numéricos.

    En el trabajo práctico, se utilizó NumPy para realizar cálculos matemáticos y manipulación de datos, como la normalización y transformación de los datos de entrada.
  ]
)

==== Pandas <pandas>

#recuadro(
  rgb("#2d18a3"),
  "Pandas",
  [
    Pandas es una biblioteca de Python que proporciona estructuras de datos y herramientas para el análisis de datos. Permite manipular y analizar datos tabulares de manera eficiente, facilitando la carga, limpieza y transformación de datos.

    En el trabajo práctico, se utilizó Pandas para cargar los datos, realizar operaciones de limpieza y transformación, así como para explorar y analizar los datos antes de entrenar los modelos.
  ]
)

==== Polars <polars>

#recuadro(
  rgb("#1681FF"),
  "Polars",
  [
    Polars es una biblioteca de análisis de datos en Python que se centra en la velocidad y la eficiencia. Utiliza un motor de ejecución paralelo y optimizado para realizar operaciones sobre grandes conjuntos de datos de manera rápida.

    En el trabajo práctico, se utilizó Polars para realizar operaciones de análisis y manipulación de datos, aprovechando su rendimiento superior en comparación con otras bibliotecas como Pandas.
  ]
)

=== Visualización de Datos <visualizacion>

==== MatPlotLib <matplotlib>

#recuadro(
  rgb("#EA7324"),
  "MatPlotLib",
  [
    Matplotlib es una biblioteca de visualización de datos en Python que permite crear gráficos estáticos, animados e interactivos. Proporciona una amplia variedad de tipos de gráficos y opciones de personalización.

    En el trabajo práctico, se utilizó Matplotlib para crear gráficos y visualizaciones de los resultados obtenidos por los modelos, facilitando la interpretación y análisis de los datos.
  ]
)

==== Seaborn <seaborn>

#recuadro(
  rgb("#89ADC8"),
  "Seaborn",
  [
    Seaborn es una biblioteca de visualización de datos basada en Matplotlib que proporciona una interfaz de alto nivel para crear gráficos atractivos y informativos. Seaborn facilita la creación de gráficos estadísticos y la visualización de relaciones entre variables.

    En el trabajo práctico, se utilizó Seaborn para crear gráficos estadísticos y visualizaciones más complejas, mejorando la presentación de los resultados.
  ]
)

==== Plotly <plotly>

#recuadro(
  rgb("#18A0FF"),
  "Plotly",
  [
    Plotly es una biblioteca de visualización interactiva que permite crear gráficos y dashboards interactivos. Proporciona una amplia variedad de tipos de gráficos y opciones de personalización.

    En el trabajo práctico, se utilizó Plotly para crear visualizaciones interactivas que permiten explorar los resultados de manera más dinámica.
  ]
)

=== Aprendizaje Automático <aprendizaje>

==== TensorFlow <tensor-flow>

#recuadro(
  rgb("#EE922B"),
  "TensorFlow",
  [
    TensorFlow es una biblioteca de código abierto para el aprendizaje automático y el aprendizaje profundo. Proporciona herramientas y recursos para construir y entrenar modelos de aprendizaje automático, así como para implementar redes neuronales profundas.

    En el trabajo práctico, se utilizó TensorFlow para construir y entrenar los modelos de aprendizaje profundo, aprovechando su flexibilidad y rendimiento.
  ]
)

==== Keras <keras>

#recuadro(
  rgb("#D20808"),
  "Keras",
  [
    Keras es una API de alto nivel para construir y entrenar modelos de aprendizaje profundo. Se integra con TensorFlow y proporciona una interfaz sencilla y fácil de usar para crear redes neuronales.

    En el trabajo práctico, se utilizó Keras para construir y entrenar los modelos de aprendizaje profundo, facilitando la implementación y ajuste de los hiperparámetros.
  ]
)

==== PyTorch <pytorch>

#recuadro(
  rgb("#F05136"),
  "PyTorch",
  [
    PyTorch es una biblioteca de aprendizaje profundo de código abierto que proporciona una interfaz flexible y dinámica para construir y entrenar modelos. Es ampliamente utilizada en la investigación y la industria debido a su facilidad de uso y rendimiento.

    En el trabajo práctico, se utilizó PyTorch para construir y entrenar los modelos de aprendizaje profundo, aprovechando su flexibilidad y capacidad para manejar datos dinámicos.
  ]
)

==== Scikit-learn <scikit-learn>

#recuadro(
  rgb("#F59C4C"),
  "Scikit-learn",
  [
    Scikit-learn es una biblioteca de aprendizaje automático en Python que proporciona herramientas para la construcción y evaluación de modelos. Incluye algoritmos de clasificación, regresión y agrupamiento, así como herramientas para la selección y evaluación de modelos.

    En el trabajo práctico, se utilizó Scikit-learn para realizar tareas de preprocesamiento de datos, selección de características y evaluación de modelos.
  ]
)

==== JAX <jax>

#recuadro(
  rgb("#087E66"),
  "JAX",
  [
    JAX es una biblioteca de Python que permite la diferenciación automática y la ejecución en GPU/TPU. Proporciona herramientas para realizar cálculos numéricos de manera eficiente y flexible.

    En el trabajo práctico, se utilizó JAX para realizar cálculos numéricos y optimización de modelos, aprovechando su capacidad para ejecutar operaciones en paralelo.
  ]
)

=== Aprendizaje por Refuerzo <aprendizaje-refuerzo>

==== Stable Baselines3 <stable-baselines3>

#recuadro(
  rgb("#F9A825"),
  "Stable Baselines3",
  [
    Stable Baselines3 es una biblioteca de aprendizaje por refuerzo en Python que proporciona implementaciones de algoritmos populares de aprendizaje por refuerzo. Facilita la construcción y entrenamiento de agentes de aprendizaje por refuerzo.

    En el trabajo práctico, se utilizó Stable Baselines3 para implementar y entrenar el modelo PPO, aprovechando su facilidad de uso y rendimiento.
  ]
)

=== Simulación de Entornos <simulacion>

==== OpenAI Gym <openai-gym>

#recuadro(
  rgb("#16AC86"),
  "OpenAI Gym",
  [
    OpenAI Gym es una biblioteca que proporciona un entorno para el desarrollo y evaluación de algoritmos de aprendizaje por refuerzo. Ofrece una variedad de entornos simulados para entrenar y evaluar agentes.

    En el trabajo práctico, se utilizó OpenAI Gym para simular el entorno del paciente y evaluar el rendimiento del modelo PPO.
  ]
)

==== TensorFlow Agents <tf-agents>

#recuadro(
  rgb("#EE922B"),
  "TensorFlow Agents",
  [
    TensorFlow Agents es una biblioteca de aprendizaje por refuerzo que proporciona herramientas y recursos para construir y entrenar agentes de aprendizaje por refuerzo en entornos simulados.

    En el trabajo práctico, se utilizó TensorFlow Agents para implementar y entrenar el modelo PPO, aprovechando su integración con TensorFlow.
  ]
)

=== Paralelismo <paralelismo>

==== Joblib <joblib>

#recuadro(
  rgb("#E25B1E"),
  "Joblib",
  [
    Joblib es una biblioteca de Python que proporciona herramientas para la paralelización y el almacenamiento en caché de funciones. Facilita la ejecución de tareas en paralelo y la gestión de recursos computacionales.

    En el trabajo práctico, se utilizó Joblib para paralelizar el entrenamiento de los modelos y optimizar el uso de recursos computacionales.
  ]
)

= Dataset <dataset>

El dataset elegido para realizar este análisis es el DiaTrend @diatrend.

#TODO(
  [
    #text(size: 25pt)[#highlight[*#underline[_Completar con el análisis hecho sobre el dataset. _]*]]
  ]
)


= Metodología de Entrenamiento <metodologia>


= Resultados y Análisis <resultados>


= Conclusiones <conclusiones>

#show bibliography: set heading(numbering: "1.")
#bibliography("bibliografia.bib", title: "Bibliografía", style: "ieee", full: true)