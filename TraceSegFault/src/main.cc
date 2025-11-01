#include <cstdint>
#include <cstdlib>
#include <iostream>

#if defined(_WIN32)
#include <process.h>
#define GETPID _getpid()
#define NOINLINE __declspec(noinline)
#else
#include <unistd.h>
#define GETPID getpid()
#define NOINLINE __attribute__((noinline))
#endif

NOINLINE void crash_here() {
    std::cout << "[crash_here] about to deref a null pointer...\n";
    volatile int* p = nullptr;       // 使用 volatile 防止被优化掉
    std::cout << "p=" << p << '\n';  // 打印一下，确认不是被优化的临时值
    *p = 42;                         // ← 这里必定 SIGSEGV
}

NOINLINE void level2() { crash_here(); }
NOINLINE void level1() { level2(); }

int main() {
    std::cout << "PID=" << GETPID << '\n';
    level1();
    return 0;
}
