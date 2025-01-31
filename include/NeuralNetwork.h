#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <stdexcept>

/**
 * @brief Enumeración de funciones de activación disponibles.
 */
enum class ActivationFunction {
    RELU,
    SIGMOID,
    TANH,
    LINEAR
};
/**
 * @brief Tipos de funciones de error disponibles
 */
enum class ErrorFunction {
    MSE,  // Error Cuadrático Medio
    MAE,  // Error Absoluto Medio
    CROSS_ENTROPY // Entropía Cruzada
};

/**
 * @class NeuralNetwork
 * @brief Clase que representa una red neuronal simple de propagación hacia adelante.
 *
 * - Permite configurar la topología (número de capas y neuronas por capa).
 * - Ofrece la posibilidad de establecer funciones de activación para capas ocultas y de salida.
 * - Dispone de un método `setWeights` para cargar pesos y sesgos entrenados externamente.
 * - El método `forward` realiza la inferencia dada una entrada.
 */
class NeuralNetwork {
public:
    /**
     * @brief Constructor de la red neuronal.
     * @param layers Vector con el número de neuronas por capa, p.ej. {4,8,3}.
     * @param hiddenAct Función de activación para las capas ocultas.
     * @param outputAct Función de activación para la capa de salida.
     */
    NeuralNetwork(const std::vector<int>& layers,
                  ActivationFunction hiddenAct = ActivationFunction::RELU,
                  ActivationFunction outputAct = ActivationFunction::SIGMOID);
                  /**
     * @brief Constructor de la red neuronal.
     * @param layers Vector con el número de neuronas por capa, p.ej. {4,8,3}.
     * @param hiddenAct Función de activación para las capas ocultas.
     * @param outputAct Función de activación para la capa de salida.
     */
    float calculateMSE(const std::vector<float>& output, const std::vector<float>& target);
    /**
     * @brief Establece manualmente los pesos y sesgos de la red.
     * @param weights Vector de matrices de pesos, aplanadas. Cada elemento es un vector que
     *        representa los pesos de una capa.
     * @param biases Vector de vectores de sesgos. Cada elemento es el sesgo para una capa.
     */
    void setWeights(const std::vector<std::vector<float>>& weights,
                    const std::vector<std::vector<float>>& biases);
    
    /**
     * 
     */
    void updateWeights(const std::vector<std::vector<float>>& gradients_weights,
                                  const std::vector<std::vector<float>>& gradients_biases,
                                  float learningRate);
    /**
     * @brief Función para calcular el error dependiendo del tipo seleccionado.
     * @param output Vector con las predicciones de la red neuronal.
     * @param target Vector con los valores esperados.
     * @param errorType Tipo de función de error (MSE, MAE o Cross-Entropy).
     * @return Valor del error calculado.
     */
    float calculateError(const std::vector<float>& output, 
                        const std::vector<float>& target, 
                        ErrorFunction errorType);

    /**
     * 
     */
    void computeGradients(const std::vector<float>& input,
                                     const std::vector<float>& target,
                                     std::vector<std::vector<float>>& grad_weights,
                                     std::vector<std::vector<float>>& grad_biases);
    /**
     * @brief Realiza la propagación hacia adelante dado un vector de entrada.
     * @param input Vector de floats correspondiente a la entrada de la red.
     * @return Vector de floats que representa la salida de la red.
     */
    std::vector<float> forward(const std::vector<float>& input);


private:
    /**
     * @brief Inicializa los pesos y sesgos con valores fijos (0.1f).
     *        Ideal para ejemplos, aunque en casos reales se recomienda
     *        inicialización aleatoria o usar `setWeights`.
     */
    void initWeights();

    // Estructura de la red: número de neuronas por capa
    std::vector<int> m_layers;

    // Para cada capa, se guardan los pesos en un vector aplanado
    // Ejemplo: la i-ésima capa almacenará (n_in * n_out) pesos
    std::vector<std::vector<float>> m_weights;

    // Para cada capa, se guarda un vector de longitud igual a n_out para los sesgos
    std::vector<std::vector<float>> m_biases;

    // Funciones de activación para capas ocultas y capa de salida
    ActivationFunction m_hiddenActivation;
    ActivationFunction m_outputActivation;
};

#endif // NEURAL_NETWORK_H
