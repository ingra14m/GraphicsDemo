#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

inline double Degree(double angle) { return angle * MY_PI / 180.0; }  // 角度制转弧度制

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos) {
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

//    Eigen::Matrix4f translate;
    view << 1, 0, 0, -eye_pos[0],
            0, 1, 0, -eye_pos[1],
            0, 0, 1, -eye_pos[2],
            0, 0, 0, 1;

//    view = translate * view;

    return view;

//    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle) {
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.

    double degree = Degree(rotation_angle);
    model << cos(degree), -sin(degree), 0, 0,
            sin(degree), cos(degree), 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;
    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar) {
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.

    // 这个推导公式在很多地方都有，总体来说可以将透视投影拆分为两部分
    // 1. 压缩成立方体
    // 2. 过一个正交投影矩阵

    // 压缩成立方体的过程需要利用到三个假设才能求出最后的参数
    // 1.近平面的所有点坐标不变
    //
    // 2.远平面的所有点坐标z值不变 都是f
    //
    // 3.远平面的中心点坐标值不变 为(0,0,f)
    float n = zNear;
    float f = zFar;
    float t = -abs(zNear) * tan(Degree(eye_fov) / 2.0);
    float r = t * aspect_ratio;

    projection << n / r, 0, 0, 0,
            0, n / t, 0, 0,
            0, 0, (n + f) / (n - f), -2 * n * f / (n - f),
            0, 0, 1, 0;

    return projection;

}

int main(int argc, const char **argv) {
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        } else
            return 0;
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};  // 定义相机位置

    std::vector<Eigen::Vector3f> pos{{2,  0, -2},
                                     {0,  2, -2},
                                     {-2, 0, -2}};  // 定义一个三角形

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    // 给光栅器传入信息
    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;  // 记录按键信息
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);  // 居然还能通过enum这么玩，清空framebuffer，深度用最大值清空

        r.set_model(get_model_matrix(angle));  // 设置M矩阵（只包含了旋转）
        r.set_view(get_view_matrix(eye_pos));  // 设置V矩阵（根据camera）
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);  // 仅仅设置了frame Buffer，设置了一条线上所有的像素。
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        // std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        } else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}
