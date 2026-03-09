#include <Python.h>
#include <iostream>
#include <string>

int main() {
    int exitCode = 0;
    PyObject *moduleName = nullptr;
    PyObject *module = nullptr;
    PyObject *addFunc = nullptr;
    PyObject *addArgs = nullptr;
    PyObject *addResult = nullptr;
    PyObject *messageFunc = nullptr;
    PyObject *messageArgs = nullptr;
    PyObject *messageResult = nullptr;
    PyObject *messageStr = nullptr;

    Py_Initialize();
    if (!Py_IsInitialized()) {
        std::cerr << "Failed to initialize the Python interpreter.\n";
        return 1;
    }

    // Make sure the current directory is importable so embedded_module.py can be found.
    PyRun_SimpleString("import sys, os\nsys.path.insert(0, os.path.abspath('.'))");

    do {
        moduleName = PyUnicode_FromString("embedded_module");
        if (!moduleName) {
            PyErr_Print();
            exitCode = 1;
            break;
        }

        module = PyImport_Import(moduleName);
        if (!module) {
            PyErr_Print();
            exitCode = 1;
            break;
        }

        addFunc = PyObject_GetAttrString(module, "add_numbers");
        if (!addFunc || !PyCallable_Check(addFunc)) {
            std::cerr << "Python function add_numbers is not available.\n";
            exitCode = 1;
            break;
        }

        addArgs = Py_BuildValue("(ii)", 6, 7);
        if (!addArgs) {
            PyErr_Print();
            exitCode = 1;
            break;
        }

        addResult = PyObject_CallObject(addFunc, addArgs);
        if (!addResult) {
            PyErr_Print();
            exitCode = 1;
            break;
        }

        long sum = PyLong_AsLong(addResult);
        if (PyErr_Occurred()) {
            PyErr_Print();
            exitCode = 1;
            break;
        }

        std::cout << "C++ received sum from Python: " << sum << std::endl;

        messageFunc = PyObject_GetAttrString(module, "format_message");
        if (!messageFunc || !PyCallable_Check(messageFunc)) {
            std::cerr << "Python function format_message is not available.\n";
            exitCode = 1;
            break;
        }

        messageArgs = Py_BuildValue("(s)", "Codex user");
        if (!messageArgs) {
            PyErr_Print();
            exitCode = 1;
            break;
        }

        messageResult = PyObject_CallObject(messageFunc, messageArgs);
        if (!messageResult) {
            PyErr_Print();
            exitCode = 1;
            break;
        }

        messageStr = PyObject_Str(messageResult);
        if (!messageStr) {
            PyErr_Print();
            exitCode = 1;
            break;
        }

        std::cout << "C++ received message: " << PyUnicode_AsUTF8(messageStr) << std::endl;
    } while (false);

    Py_XDECREF(messageStr);
    Py_XDECREF(messageResult);
    Py_XDECREF(messageArgs);
    Py_XDECREF(messageFunc);
    Py_XDECREF(addResult);
    Py_XDECREF(addArgs);
    Py_XDECREF(addFunc);
    Py_XDECREF(module);
    Py_XDECREF(moduleName);

    if (Py_IsInitialized()) {
        Py_Finalize();
    }

    return exitCode;
}
