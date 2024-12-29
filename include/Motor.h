#ifndef MOTOR_H
#define MOTOR_H

#include <Arduino.h>

class Motor {
private:
    int pin;
public:
    Motor(int pin);
    void setEstado(bool estado);
};

#endif // MOTOR_H
