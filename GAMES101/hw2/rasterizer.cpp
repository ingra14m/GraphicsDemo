#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions) {
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices) {
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols) {
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}


auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f) {
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

// v是一个Vector3f的数组，数组大小为3，表示vertex
static bool insideTriangle(float x, float y, const Vector3f *_v) {
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    // get the vectices of the triangle
    const Vector3f &A = _v[0];
    const Vector3f &B = _v[1];
    const Vector3f &C = _v[2];
    // get the line of the triangle
    Vector3f AB = B - A;
    Vector3f BC = C - B;
    Vector3f CA = A - C;
    // use cross multiply to decide whether it is inside
    Vector3f P(x, y, 1.0);
    Vector3f AP = P - A;
    Vector3f BP = P - B;
    Vector3f CP = P - C;

    Vector3f res_1 = AB.cross(AP);
    Vector3f res_2 = BC.cross(BP);
    Vector3f res_3 = CA.cross(CP);

    // if res_1,2,3 point to the same direction in z, it is inside
    return (res_1.z() > 0 && res_2.z() > 0 && res_3.z() > 0) || (res_1.z() < 0 && res_2.z() < 0 && res_3.z() < 0);
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f *v) {
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) /
               (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() -
                v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) /
               (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() -
                v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) /
               (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() -
                v[1].x() * v[0].y());
    return {c1, c2, c3};
}

// 这里还是只是设置了三角形的vertex + color
void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type) {
    auto &buf = pos_buf[pos_buffer.pos_id];
    auto &ind = ind_buf[ind_buffer.ind_id];
    auto &col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto &i : ind) {
        Triangle t;
//        std::cout << view << std::endl;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };


        //Homogeneous division
        for (auto &vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto &vert : v) {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i) {
            t.setVertex(i, v[i].head<3>());
//            t.setVertex(i, v[i].head<3>());
//            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle &t) {
    auto v = t.toVector4();

    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle
    float x_min = std::min(std::min(v[0][0], v[1][0]), v[2][0]);
    float x_max = std::max(std::max(v[0][0], v[1][0]), v[2][0]);
    float y_min = std::min(std::min(v[0][1], v[1][1]), v[2][1]);
    float y_max = std::max(std::max(v[0][1], v[1][1]), v[2][1]);

    // anti-alising
    bool MSAA4X = true;

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
    if (!MSAA4X) {
        // without anti-alising
        for (int x = (int) x_min; x <= (int) x_max; x++) {
            for (int y = (int) y_min; y <= (int) y_max; y++) {
                auto sampled_x = float(x) + 0.5;
                auto sampled_y = float(y) + 0.5;
                // we need to decide whether this point is actually inside the triangle
//                if (!insideTriangle((float) x, (float) y, t.v)) continue;  // 这个对应的是左下角，我们采用下一行的像素中心点来判断
                if (!insideTriangle(sampled_x, sampled_y, t.v)) continue;
                // get z value--depth
                // If so, use the following code to get the interpolated z value.
                auto[alpha, beta, gamma] = computeBarycentric2D(sampled_x, sampled_y, t.v);
                float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated =
                        alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;

                // compare the current depth with the value in depth buffer
                if (depth_buf[get_index(x, y)] >
                    z_interpolated)// note: we use get_index to get the index of current point in depth buffer
                {
                    // we have to update this pixel
                    depth_buf[get_index(x, y)] = z_interpolated; // update depth buffer
                    // assign color to this pixel
                    set_pixel(Vector3f(x, y, z_interpolated), t.getColor());
                }
            }
        }
    } else {
        for (int x = (int) x_min; x <= (int) x_max; x++) {
            for (int y = (int) y_min; y <= (int) y_max; y++) {
                // you have to record the min-depth of the 4 sampled points(in one pixel)
                float min_depth = FLT_MAX;
                // the number of the 4 sampled points that are inside triangle
                int count = 0;
                std::vector<std::vector<float>> sampled_points{{0.25, 0.25},
                                                               {0.25, 0.75},
                                                               {0.75, 0.25},
                                                               {0.75, 0.75}};
                for (int i = 0; i < 4; i++) {
                    auto sampled_x = float(x) + sampled_points[i][0];
                    auto sampled_y = float(y) + sampled_points[i][1];
                    if (insideTriangle(sampled_x, sampled_y, t.v)) {
                        auto[alpha, beta, gamma] = computeBarycentric2D(sampled_x, sampled_y, t.v);
                        float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                        float z_interpolated =
                                alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                        z_interpolated *= w_reciprocal;
                        min_depth = std::min(min_depth, z_interpolated);
                        count += 1;
                    }
                }
                if (count > 0 && depth_buf[get_index(x, y)] > min_depth) {
                    // update
                    depth_buf[get_index(x, y)] = min_depth;
                    // note: the color should be changed too
                    set_pixel(Vector3f(x, y, min_depth), t.getColor() * count / 4.0 +
                                                         frame_buf[get_index(x, y)] * (4 - count) /
                                                         4.0); // frame_buf contains the current color
                }
            }
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f &m) {
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f &v) {
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f &p) {
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff) {
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color) {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth) {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h) {
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y) {
    return (height - 1 - y) * width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f &point, const Eigen::Vector3f &color) {
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height - 1 - point.y()) * width + point.x();
    frame_buf[ind] = color;

}

// clang-format on