#include <iostream>
#include <vector>

#include "shitfilter.hpp"

// ===================== Demo ===================== //
int main() {
    std::vector<double> sample = {10, 11, 12, 100, 13, 12, 11, 150, 12, 11, 10};

    GradientMeanFilter gm(3);  // 窗口=3，默认其余参数

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Push & filter online:\n";
    for (double x : sample) {
        double y = gm.push(x);
        std::cout << "  in=" << x << "  ->  out=" << y << '\n';
    }

    // 若想拿整段结果
    const auto& full = gm.getFiltered();
    std::cout << "\n=== Summary ===\nOriginal : ";
    for (double x : sample) std::cout << x << ' ';
    std::cout << "\nFiltered : ";
    for (double y : full) std::cout << y << ' ';
    std::cout << '\n';

    // 试试 reset + 新参数
    gm.reset(5, 0.05, 80, 2.0);  // window=5, lr=0.05, epochs=80, k=2.0
    for (double x : sample) gm.push(x);
    std::cout << "After reset(window=5) : ";
    for (double y : gm.getFiltered()) std::cout << y << ' ';
    std::cout << '\n';
}
