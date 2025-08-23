#include <iostream>

#include "pid.hpp"

int main() {
    PID_param pid_param(10, 0, 0, std::pair<double, double>(-20, 40),
                        std::pair<double, double>(-10, 10), 0.0);
    PID_Controller pid(pid_param);

    std::cout << "Hell Here" << std::endl;
    return 0;
}
