# vehiculo_con_red_neuronal

# Motor de Inferencia en C++

Este repositorio contiene un **motor de inferencia** que permite ejecutar (o "inferir") una red neuronal *feedforward* en C++.  
Está diseñado de manera sencilla para que pueda ser utilizado tanto por **personal técnico** de forma sencilla
---

## 1. ¿Qué es y para qué sirve?

Este motor de inferencia recibe un **modelo** (definido por sus capas y pesos) y, dada una **entrada** en forma de vector de floats, produce una **salida**.  
Su objetivo principal es tomar el trabajo de un modelo previamente entrenado (en otra plataforma o en el mismo entorno) y realizar **predicciones** de manera sencilla.

### Características principalesword

1. **Configurable**: Puedes definir el número de capas y de neuronas por capa.  
2. **Funciones de activación**: Ofrece las más comunes (RELU, SIGMOID, TANH y LINEAR).  
3. **Asignación de pesos**: Permite inicializar pesos manualmente (por ejemplo, tras haber entrenado en otra herramienta como TensorFlow o PyTorch).  
4. **Inferencia sencilla**: Usa la función `forward()` para hacer la propagación de las entradas hasta la salida.

---

## 2. Estructura del Código

El componente principal es la **clase** `NeuralNetwork`.  
Se construye así:

1. **Enumeración `ActivationFunction`**  
   Define los tipos de funciones de activación disponibles:
   ```cpp
   enum class ActivationFunction {
       RELU,
       SIGMOID,
       TANH,
       LINEAR
   };
   ```

2. **Funciones de activación**  
   - `relu(x)`  
   - `sigmoid(x)`  
   - `tanh_custom(x)`  
   - `linear(x)`  

   Y un **"helper"** `applyActivation(float x, ActivationFunction func)` que selecciona la activación correcta según el valor del enum.

3. **Clase `NeuralNetwork`**  
   - **Constructor**:  
     Recibe un vector `layers` (por ejemplo, `{4, 8, 3}`) que indica cuántas **neuronas** habrá en cada capa.  
     Además, recibe dos funciones de activación, una para las capas "ocultas" (`hiddenAct`) y otra para la capa de salida (`outputAct`).  
     ```cpp
     NeuralNetwork(const std::vector<int>& layers,
                   ActivationFunction hiddenAct,
                   ActivationFunction outputAct);
     ```
   - **Método `initWeights()`**:  
     Inicializa los pesos y sesgos con un valor fijo (0.1f) a modo de ejemplo.
   - **Método `setWeights(...)`**:  
     Permite asignar pesos y sesgos entrenados externamente.  
   - **Método `forward(...)`**:  
     Ejecuta el **proceso de inferencia** aplicando multiplicaciones de matrices, sumas de sesgos y la **función de activación** capa a capa.

4. **Ejemplo en `main()`**  
   Se muestra cómo crear una instancia de la red, asignarle un tamaño de capas, y usar `forward()` para obtener la salida.

---

#

### B. Integración en otro proyecto C++

1. Copia el código en tu proyecto (idealmente separando el `.h` y el `.cpp`, si lo deseas).  
2. Incluye el header donde necesites usar la red:
   ```cpp
   #include "NeuralNetwork.h"
   ```
3. Crea la red, pasa la topología (capas) y la función de activación deseada.
4. Llama a `forward()` con tu vector de entrada.

---

## 4. Personalización de la Red

1. **Definir la topología**  
   Decide cuántas **capas** y cuántas **neuronas** por capa.  
   ```cpp
   // Red con 4 neuronas de entrada, 8 en la capa oculta y 3 salidas
   std::vector<int> layers = {4, 8, 3};
   NeuralNetwork nn(layers, ActivationFunction::RELU, ActivationFunction::SIGMOID);
   ```

2. **Asignar pesos (modelo entrenado)**  
   Tras entrenar un modelo en otra herramienta, puedes exportar sus pesos y sesgos para usarlos aquí. Debes cargarlos así:
   ```cpp
   std::vector<std::vector<float>> myWeights; // dimensionado: (layers.size() - 1) vectores
   std::vector<std::vector<float>> myBiases;
   // Llenar myWeights[i] y myBiases[i] con tus datos entrenados
   nn.setWeights(myWeights, myBiases);
   ```

3. **Cambiar la función de activación**  
   Por defecto, en el constructor se ofrece un **par** de funciones:  
   - `hiddenAct`: usada en todas las capas **menos** la última.  
   - `outputAct`: usada en la última capa (capa de salida).  
   Disponibles: `RELU`, `SIGMOID`, `TANH`, `LINEAR`.

4. **Escalado de entrada**  
   Asegúrate de enviar valores en los rangos esperados. Si tu red fue entrenada con valores normalizados, debes normalizar tus datos de entrada antes de llamar `forward()`.

---

## 5. Ejemplo de Uso

Aquí una pequeña parte del `main()` para ilustrarlo:

```cpp
// Definimos la topología: {4, 8, 3} (4 entradas, 8 ocultas, 3 salidas).
std::vector<int> layerSizes = {4, 8, 3};

// Creamos la red, con ReLU en capas ocultas y Sigmoid en la de salida
NeuralNetwork nn(layerSizes, ActivationFunction::RELU, ActivationFunction::SIGMOID);

// Preparamos un vector de entrada con 4 valores
std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};

// Realizamos la inferencia
std::vector<float> output = nn.forward(input);

// Mostramos la salida
std::cout << "Salidas de la red: ";
for (auto val : output) {
    std::cout << val << " ";
}
std::cout << std::endl;
```

---

## 6. Preguntas Frecuentes

1. **¿Puedo añadir más capas ocultas?**  
   Sí, solo aumenta la longitud del `std::vector<int>`. Ejemplo: `{4, 8, 8, 3}` añade una capa oculta extra.

2. **¿Por qué los pesos están inicializados en 0.1?**  
   Es un ejemplo. Se recomienda usar `setWeights()` para cargar pesos reales.

3. **¿Soporta entrenamiento?**  
   No, este ejemplo **solo** cubre la parte de **inferencia** (no incluye retropropagación).

4. **¿Puedo usar funciones de activación distintas por capa?**  
   Con esta versión, tienes una para capas ocultas y otra para salida. Para personalizar cada capa por separado, se debería extender el código y asignar una función de activación distinta en cada conexión.

