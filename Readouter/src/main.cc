#include <chrono>
#include <iostream>
#include <queue>
#include <string>
#include <thread>

std::queue<std::string> str_queue = std::queue<std::string>();

int main(int argc, char* argv[]) {
    while (true) {
        std::string input;
        std::cout << "\nEnter a string > ";
        std::getline(std::cin, input);

        str_queue.push(input);

        if (str_queue.empty()) {
            continue;
        }

        std::string str_print = str_queue.front();
        for (char& c : str_print) {
            switch (c) {
                case ' ': {
                    std::cout << c << std::flush;
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    break;
                }
                case '\n': {
                    std::cout << c << std::flush;
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    break;
                }
                default: {
                    std::cout << c << std::flush;
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            }
        }
        str_queue.pop();
    }

    return 0;
}
