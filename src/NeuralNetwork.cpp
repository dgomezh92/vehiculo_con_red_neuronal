#include "NeuralNetwork.h"

// Función de activación ReLU (Rectified Linear Unit)
// Esta función toma un valor de entrada 'x' y devuelve el mayor valor entre 0 y 'x',
// utilizando la fórmula matemática:
// relu(x) = max(0, x)
//
// Propiedades:
// - Si x es positivo o cero, la salida será igual a x.
// - Si x es negativo, la salida será 0.
// - Es una función no lineal que introduce esparsidad en las activaciones de una red neuronal.
//
// Usos comunes en redes neuronales:
// - Función de activación estándar en capas ocultas, debido a su simplicidad y buen rendimiento.
// - Ayuda a mitigar el problema del desvanecimiento del gradiente (vanishing gradient).
//
// Ejemplos de salida:
// - relu(5) = 5
// - relu(0) = 0
// - relu(-3) = 0
//
// Ventajas:
// - Computacionalmente eficiente (simple comparación).
// - Introduce no linealidad, permitiendo a la red aprender relaciones complejas.
// 
// Limitaciones:
// - "Dying ReLU": Si muchas entradas son negativas, algunas neuronas pueden quedar inactivas 
//   (es decir, siempre producirán 0 como salida).
float NeuralNetwork::relu(float x) {
    return fmax(0, x);
}

// Función de activación Sigmoid
// Esta función toma un valor de entrada 'x' y lo transforma a un rango entre 0 y 1,
// utilizando la fórmula matemática:
// sigmoid(x) = 1 / (1 + exp(-x))
// 
// Propiedades:
// - Devuelve un valor suavizado en el rango (0, 1), ideal para modelar probabilidades.
// - Si x es grande (positivo), la salida tiende a 1.
// - Si x es pequeño (negativo), la salida tiende a 0.
// - Si x es 0, la salida será 0.5 (el punto medio).
//
// Usos comunes en redes neuronales:
// - Normalizar la salida de una capa para que esté entre 0 y 1.
// - Como función de activación en problemas de clasificación binaria.
// 
// Ejemplos de salida:
// - sigmoid(0) = 0.5
// - sigmoid(2) ≈ 0.88
// - sigmoid(-3) ≈ 0.047
float NeuralNetwork::sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Método predict
void NeuralNetwork::predict(float input1, float input2, float duracion1, float duracion2, float outputs[4]) {
    float layer1[8];
    for (int i = 0; i < 8; i++) {
        layer1[i] = input1 * PESOS_CAPA_0[i] + input2 * PESOS_CAPA_0[i + 8] +
                    duracion1 * PESOS_CAPA_0[i + 16] + duracion2 * PESOS_CAPA_0[i + 24] +
                    SESGOS_CAPA_0[i];
        layer1[i] = relu(layer1[i]); // Uso de ReLU
    }

    for (int i = 0; i < 4; i++) {
        outputs[i] = 0;
        for (int j = 0; j < 8; j++) {
            outputs[i] += layer1[j] * PESOS_CAPA_1[j * 4 + i];
        }
        outputs[i] += SESGOS_CAPA_1[i];
        outputs[i] = sigmoid(outputs[i]); // Uso de Sigmoid
    }
}
