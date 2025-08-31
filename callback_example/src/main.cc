#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

// 全局同步工具
std::mutex mtx;
std::condition_variable cv;
bool ready = false;

// 回调函数类型
using Callback = std::function<void()>;

// 工作线程函数，等待回调触发
void worker_thread(Callback cb) {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [] { return ready; });  // 等待 ready == true
    lock.unlock();

    // 调用传入的回调函数
    cb();
}

int main() {
    // 定义一个回调
    Callback cb = []() { std::cout << "Worker thread: Callback triggered!" << std::endl; };

    // 启动工作线程，等待回调触发
    std::thread t(worker_thread, cb);

    // 主线程睡眠模拟某些工作
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Main thread: now triggering callback..." << std::endl;

    // 唤醒线程
    {
        std::lock_guard<std::mutex> lock(mtx);
        ready = true;
    }
    cv.notify_one();

    t.join();
    return 0;
}
