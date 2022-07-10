#pragma once

#include <vector>
#include <memory>
#include "Vector.hpp"
#include "Object.hpp"
#include "Light.hpp"

class Scene {
public:
    // setting up options
    int width = 1280;
    int height = 960;
    double fov = 90;
    Vector3f backgroundColor = Vector3f(0.235294, 0.67451, 0.843137);
    int maxDepth = 5;
    float epsilon = 0.00001;

    Scene(int w, int h) : width(w), height(h) {}

//    void Add(std::unique_ptr <Object> object) { objects.push_back(std::move(object)); }
    // 这里不能用引用来接收，因为unique_ptr的问题。unique_ptr想push_back只能借助右值，要么右值传入，要么直接构造make_unique
    void Add(std::unique_ptr<Object> object) { objects.push_back(std::move(object)); }

    // Light的传入参数为Add(make_unique(ddd))，会直接构建在light，如果创建好的aa传入会拷贝,unique_ptr无法过拷贝，可以借助左值
    void Add(std::unique_ptr <Light> light) { lights.push_back(std::move(light)); }


    [[nodiscard]] const std::vector <std::unique_ptr<Object>> &get_objects() const { return objects; }

    [[nodiscard]] const std::vector <std::unique_ptr<Light>> &get_lights() const { return lights; }

private:
    // creating the scene (adding objects and lights)
    std::vector <std::unique_ptr<Object>> objects;
    std::vector <std::unique_ptr<Light>> lights;
};