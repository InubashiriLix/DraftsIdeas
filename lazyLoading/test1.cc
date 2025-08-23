#include <iostream>
#include <memory>

class Expensive {
   public:
    Expensive() { std::cout << "doing something to the Exp" << std::endl; }
    void doSomething() { std::cout << "doing something" << std::endl; }
};

class LazyLoader {
   private:
    std::unique_ptr<Expensive> resource;

   public:
    Expensive* getResource() {
        if (!resource) {
            resource = std::make_unique<Expensive>();
        }
        return resource.get();
    }
};

int main() {
    LazyLoader loader = LazyLoader();
    loader.getResource()->doSomething();
    return 0;
}
