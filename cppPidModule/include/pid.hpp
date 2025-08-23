#pragma once

#include <algorithm>
#include <chrono>
#include <stdexcept>

struct PID_param {
    double Kp;
    double Kd;
    double Ki;
    std::pair<double, double> o_max;
    std::pair<double, double> i_max;
    double deadzone;

    PID_param(double p, double d, double i, std::pair<double, double> omax,
              std::pair<double, double> imax, double deadzone) {
        if (p < 0 || d < 0 || i < 0) {
            throw std::invalid_argument("PID parameters must be non-negative");
        }

        if (omax.first > omax.second) {
            throw std::invalid_argument("Output max must be in ascending order");
        }
        if (imax.first > imax.second) {
            throw std::invalid_argument("Integral max must be in ascending order");
        }
        this->Kp = p;
        this->Kd = d;
        this->Ki = i;
        this->o_max = omax;
        this->i_max = imax;
        this->deadzone = deadzone;
    }
};

class PID_Controller {
   public:
    PID_Controller(const PID_param &params);

    void setTarget(double target);

    void reset();

    double update(double measurement, double dt = -1.0);

   private:
    PID_param param__;

    double target__ = 0.0;
    double last_error__ = 0.00;
    double integral__ = 0.00;
    std::chrono::steady_clock::time_point last_time__;
};
