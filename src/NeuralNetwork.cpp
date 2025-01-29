#include "NeuralNetwork.h"
#include <cmath>
#include <stdexcept>

/**
 * @brief Función ReLU (Rectified Linear Unit).
 */
static inline float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

/**
 * @brief Función Sigmoid.
 */
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * @brief Función Tanh.
 */
static inline float tanh_custom(float x) {
    return std::tanh(x);
}

/**
 * @brief Función Linear.
 */
static inline float linear(float x) {
    return x;
}

/**
 * @brief Helper para aplicar la función de activación según el enum.
 */
static inline float applyActivation(float x, ActivationFunction func) {
    switch (func) {
        case ActivationFunction::RELU:
            return relu(x);
        case ActivationFunction::SIGMOID:
            return sigmoid(x);
        case ActivationFunction::TANH:
            return tanh_custom(x);
        case ActivationFunction::LINEAR:
        default:
            return linear(x);
    }
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers,
                             ActivationFunction hiddenAct,
                             ActivationFunction outputAct)
    : m_layers(layers),
      m_hiddenActivation(hiddenAct),
      m_outputActivation(outputAct)
{
    // Validación mínima: se necesitan al menos 2 capas (entrada y salida)
    if (m_layers.size() < 2) {
        throw std::runtime_error("Se requieren al menos 2 capas (entrada y salida).");
    }

    // Inicializamos los pesos y sesgos con valores fijos (0.1f)
    initWeights();
}

void NeuralNetwork::initWeights() {
    m_weights.resize(m_layers.size() - 1);
    m_biases.resize(m_layers.size() - 1);

    for (size_t layerIndex = 0; layerIndex < m_layers.size() - 1; layerIndex++) {
        int inSize  = m_layers[layerIndex];
        int outSize = m_layers[layerIndex + 1];

        // Tamaño de la matriz (aplanada) de pesos
        m_weights[layerIndex].resize(inSize * outSize, 0.1f);

        // Tamaño de sesgos
        m_biases[layerIndex].resize(outSize, 0.1f);
    }
}

void NeuralNetwork::setWeights(const std::vector<std::vector<float>>& weights,
                               const std::vector<std::vector<float>>& biases)
{
    if (weights.size() != m_layers.size() - 1 ||
        biases.size()  != m_layers.size() - 1)
    {
        throw std::runtime_error("Pesos o sesgos no coinciden con la topología de la red.");
    }

    // Asigna los pesos y sesgos externos
    m_weights = weights;
    m_biases  = biases;
}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) {
    // Validamos que la entrada tenga el tamaño de la capa de entrada
    if (input.size() != static_cast<size_t>(m_layers[0])) {
        throw std::runtime_error("El vector de entrada no coincide con la capa de entrada.");
    }

    // Vector que almacena las activaciones de la capa actual
    std::vector<float> activations = input;

    // Recorremos cada capa, salvo la última (que no tiene pesos salientes)
    for (size_t layerIndex = 0; layerIndex < m_layers.size() - 1; layerIndex++) {
        int inSize  = m_layers[layerIndex];
        int outSize = m_layers[layerIndex + 1];

        std::vector<float> newActivations(outSize, 0.0f);

        // Multiplicación activations x pesos
        for (int j = 0; j < outSize; j++) {
            for (int i = 0; i < inSize; i++) {
                newActivations[j] += activations[i] * m_weights[layerIndex][i * outSize + j];
            }
            // Suma de sesgos
            newActivations[j] += m_biases[layerIndex][j];

            // Determinamos si es una capa oculta o la capa final
            ActivationFunction currentAct = (layerIndex < (m_layers.size() - 2))
                                            ? m_hiddenActivation
                                            : m_outputActivation;

            // Aplicamos la función de activación
            newActivations[j] = applyActivation(newActivations[j], currentAct);
        }

        // Actualizamos las activaciones para la siguiente capa
        activations = newActivations;
    }

    return activations;
}
