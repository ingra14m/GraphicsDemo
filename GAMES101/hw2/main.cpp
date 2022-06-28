// clang-format off
#include <iostream>
#include <opencv2/opencv.hpp>
#include "rasterizer.hpp"
#include "Triangle.hpp"

/*
 * 整个hw2的流程是
 *
 * 1. 准备阶段
 * - 准备三角形的信息，如顶点位置，顶点颜色，顶点的索引
 * - 给光栅器传入vertex、index、color
 * - 指定光栅器frame buffer大小
 * - 给光栅器传入mvp矩阵
 *
 * 2. 绘制阶段
 * - 光栅器调用draw
 * - draw每次以三角形为单位，计算出要光栅化的三角形信息，存到Triangle实例中
 * - 计算AABB中像素的情况，判断像素中心点是否在三角形内部（叉乘）（以左下角为00）
 * - 如果在内部，计算深度（需要经过投影矩阵插值），如果可以则更新framebuffer中的颜色（这里对颜色没有插值）
 *
 * 3. 代码原则
 * - 离散化的东西只是作为序号
 * - 插值计算深度的时候则是计算采样点的深度。用采样点的深度来带遍当前像素的深度。
 */

constexpr double MY_PI = 3.1415926;

inline double Degree(double angle) { return angle * MY_PI / 180.0; }

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos) {
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0],
            0, 1, 0, -eye_pos[1],
            0, 0, 1, -eye_pos[2],
            0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle) {
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    return model;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar) {
    // TODO: Copy-paste your implementation from the previous assignment.
    Eigen::Matrix4f projection;

    float n = zNear;
    float f = zFar;
    float t = -abs(zNear) * tan(Degree(eye_fov) / 2.0); // you do not need to add "-" here, it is just for visualizatin
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

    if (argc == 2) {
        command_line = true;
        filename = std::string(argv[1]);
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};


    std::vector<Eigen::Vector3f> pos
            {
                    {2,   0,   -2},
                    {0,   2,   -2},
                    {-2,  0,   -2},
                    {3.5, -1,  -5},
                    {2.5, 1.5, -5},
                    {-1,  0.5, -5}
            };

    std::vector<Eigen::Vector3i> ind
            {
                    {0, 1, 2},
                    {3, 4, 5}
            };

    std::vector<Eigen::Vector3f> cols
            {
                    {217.0, 238.0, 185.0},
                    {217.0, 238.0, 185.0},
                    {217.0, 238.0, 185.0},
                    {185.0, 217.0, 238.0},
                    {185.0, 217.0, 238.0},
                    {185.0, 217.0, 238.0}
            };

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);
    auto col_id = r.load_colors(cols);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::imshow("image", image);
        cv::imwrite("resutl.png", image);
        key = cv::waitKey(10);

        // std::cout << "frame count: " << frame_count++ << '\n';
    }

    return 0;
}
// clang-format on