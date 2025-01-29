#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

// Posibles funciones de activación
enum class ActivationFunction {
    RELU,
    SIGMOID,
    TANH,
    LINEAR
};

// --- Funciones de activación ----
inline float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float tanh_custom(float x) {
    return std::tanh(x);
}

inline float linear(float x) {
    return x;
}

// Helper para mapear el enum a la función de activación
inline float applyActivation(float x, ActivationFunction func) {
    switch (func) {
        case ActivationFunction::RELU:    return relu(x);
        case ActivationFunction::SIGMOID: return sigmoid(x);
        case ActivationFunction::TANH:    return tanh_custom(x);
        case ActivationFunction::LINEAR:  return linear(x);
        default:                          return x; // caso por defecto
    }
}

/**
 * @class NeuralNetwork
 * @brief Clase para una red neuronal feedforward parametrizable.
 *
 * Permite especificar:
 *  - Número de capas y neuronas por capa.
 *  - Funciones de activación distintas para capas ocultas y salida.
 *  - Inicializar pesos y sesgos manual o aleatoriamente.
 */
class NeuralNetwork {
public:
    /**
     * @brief Constructor
     * @param layers Vector con el número de neuronas en cada capa.
     *        Ejemplo: {4, 8, 3} => 4 entradas, 8 neuronas ocultas, 3 salidas.
     * @param hiddenAct Función de activación para capas ocultas.
     * @param outputAct Función de activación para la capa de salida.
     * @throw std::runtime_error Si no hay al menos 2 capas (entrada y salida).
     */
    NeuralNetwork(const std::vector<int>& layers,
                  ActivationFunction hiddenAct = ActivationFunction::RELU,
                  ActivationFunction outputAct = ActivationFunction::SIGMOID);

    /**
     * @brief Permite cargar pesos y sesgos externos (por ejemplo, de un modelo entrenado).
     * @param weights m_weights[i] tendrá dimensión (layers[i] * layers[i+1]).
     * @param biases m_biases[i] tendrá dimensión (layers[i+1]).
     * @throw std::runtime_error si las dimensiones no coinciden con la topología.
     */
    void setWeights(const std::vector<std::vector<float>>& weights,
                    const std::vector<std::vector<float>>& biases);

    /**
     * @brief Realiza la propagación hacia adelante (forward) con una entrada.
     * @param input Vector de tamaño layers[0].
     * @return Vector de salidas (tamaño layers.back()).
     * @throw std::runtime_error si el tamaño de input no coincide con la capa de entrada.
     */
    std::vector<float> forward(const std::vector<float>& input);

private:
    /**
     * @brief Inicializa los vectores de pesos y sesgos con un valor fijo (0.1f) 
     *        como ejemplo simple. En la práctica, se puede cambiar a inicialización
     *        aleatoria o cargar valores entrenados.
     */
    void initWeights();

private:
    std::vector<int> m_layers;                      ///< n_i: neuronas por capa
    std::vector<std::vector<float>> m_weights;      ///< Pesos por capa (tamaño = layers.size()-1)
    std::vector<std::vector<float>> m_biases;       ///< Sesgos por capa (tamaño = layers.size()-1)

    ActivationFunction m_hiddenActivation;          ///< Activación en capas ocultas
    ActivationFunction m_outputActivation;          ///< Activación en capa de salida
};

#endif // NEURALNETWORK_H
