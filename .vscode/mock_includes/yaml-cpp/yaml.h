#pragma once
// Mock yaml.h for IntelliSense
#include <string>
#include <vector>
namespace YAML {
    class Node {
    public:
        template<typename T> T as() const { return T(); }
        Node operator[](const char*) const { return Node(); }
        // Iterator support
        struct iterator {
            Node operator*() const { return Node(); }
            bool operator!=(const iterator&) const { return false; }
            iterator& operator++() { return *this; }
        };
        iterator begin() const { return iterator(); }
        iterator end() const { return iterator(); }
    };
    static Node LoadFile(const std::string&) { return Node(); }
}
