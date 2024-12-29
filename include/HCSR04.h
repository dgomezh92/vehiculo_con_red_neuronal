#ifndef HCSR04_H
#define HCSR04_H

#include <Arduino.h>

class HCSR04 {
private:
    int trigPin, echoPin;
public:
    HCSR04(int trigPin, int echoPin);
    float medirDistancia();
};

#endif // HCSR04_H
