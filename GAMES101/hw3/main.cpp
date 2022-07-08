#include <iostream>
#include <opencv2/opencv.hpp>

#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"  // 实现集成在.h文件了，有点离谱，这样会导致最终的可执行文件很大吧


/*
 * hw3的整体思路
 * 首先从obj文件中读取三角形的数据，如pos、normal、texcoord。需要注意的是，obj中的坐标都是在模型空间下，法线贴图很蓝的都是定义在切线空间。
 *      - 法线默认都是从-1到1的，与OpenGL中0-1的区间并不吻合，因此我们需要(x + 1) / 2来映射。
 * 然后之后的流程没啥不一样，将三角形的信息完整地传给光栅器
 * 在光栅器中进行mvp变换得到NDC，用于后续通过深度进行插值
 * 至于normal，在UnityShader中推导过，其变换矩阵很神奇，不能直接套用mvp，inverse + transpose才是normal的变换矩阵
 * 本次统一将所有的计算放在了view空间下，感觉有点拖沓
 *
 * 在fragment shader中 light的坐标和eye的坐标其实有点小问题，因为应该是view空间，但是这个没必要与mvp变换中的一致，知道是view空间下就可以啦
 *
 */

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

Eigen::Matrix4f get_model_matrix(float angle) {
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
            0, 1, 0, 0,
            -sin(angle), 0, cos(angle), 0,
            0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
            0, 2.5, 0, 0,
            0, 0, 2.5, 0,
            0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar) {
    // TODO: Use the same projection matrix from the previous assignments
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.

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

Eigen::Vector3f vertex_shader(const vertex_shader_payload &payload) {
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload &payload) {
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f &vec, const Eigen::Vector3f &axis) {
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light {
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload &payload) {
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture) {
        // TODO: Get the texture value at the texture coordinates of the current fragment
        // without interpolation
        return_color = payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y());
        // with interpolation
        return_color = payload.texture->getColorBilinear(payload.tex_coords.x(), payload.tex_coords.y());  // 获取纹理贴图
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;  // 使用了贴图，不是法线贴图（定义在切线空间）
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20,  20,  20},
                    {500, 500, 500}};
    auto l2 = light{{-20, 20,  0},
                    {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto &light : lights) {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        Eigen::Vector3f l = light.position - point; // view_position -> light source
        Eigen::Vector3f v = eye_pos - point; // view_position -> eye

        float r2 = l.dot(l); // l cannot be normalized before, because r2 is the distance

        // note: l, v should be normalized
        l = l.normalized();
        v = v.normalized();

        // note: ka, kd, ks are scalars in lecture, but they are vectors here, so you should use cwiseProduct
        // ambient La=ka*Ia
        Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity); // cwiseProduct--dot product

        // diffuse Ld=kd(I/r^2)max(0, nl)
        Eigen::Vector3f Ld = kd.cwiseProduct(light.intensity / r2) * std::max(0.0f, normal.normalized().dot(l));

        // specular Ls=ks(I/r^2)max(0, nh)^p
        Eigen::Vector3f h = (l + v).normalized();
        Eigen::Vector3f Ls =
                ks.cwiseProduct(light.intensity / r2) * std::pow(std::max(0.0f, normal.normalized().dot(h)), p);

        result_color += (La + Ld + Ls);
    }

    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload &payload) {
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;  // 顶点的颜色进行插值得到的是diffuse的反射率
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20,  20,  20},
                    {500, 500, 500}};
    auto l2 = light{{-20, 20,  0},
                    {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;  // from object coordinate to view coordinate
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto &light : lights) {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        Eigen::Vector3f l = light.position - point; // view_position -> light source
        Eigen::Vector3f v = eye_pos - point; // view_position -> eye

        float r2 = l.dot(l); // l cannot be normalized before, because r2 is the distance

        // note: l, v should be normalized
        l = l.normalized();
        v = v.normalized();

        // note: ka, kd, ks are scalars in lecture, but they are vectors here, so you should use cwiseProduct
        // ambient La=ka*Ia
        Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity); // cwiseProduct--dot product

        // diffuse Ld=kd(I/r^2)max(0, nl)
        Eigen::Vector3f Ld = kd.cwiseProduct(light.intensity / r2) * std::max(0.0f, normal.normalized().dot(l));

        // specular Ls=ks(I/r^2)max(0, nh)^p
        Eigen::Vector3f h = (l + v).normalized();
        Eigen::Vector3f Ls =
                ks.cwiseProduct(light.intensity / r2) * std::pow(std::max(0.0f, normal.normalized().dot(h)), p);

        result_color += (La + Ld + Ls);
    }
    return result_color * 255.f;
}


Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload &payload) {

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20,  20,  20},
                    {500, 500, 500}};
    auto l2 = light{{-20, 20,  0},
                    {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;

    // TODO: Implement displacement mapping here

    // Let n = normal = (x, y, z)
    auto x = normal.x(), y = normal.y(), z = normal.z();
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    Eigen::Vector3f t(x * y / std::sqrt(x * x + z * z), std::sqrt(x * x + z * z), z * y / sqrt(x * x + z * z));
    // Vector b = n cross product t
    Eigen::Vector3f b = normal.cross(t);
    // Matrix TBN = [t b n]
    Eigen::Matrix3f TBN;
    TBN << t.x(), b.x(), normal.x(),
            t.y(), b.y(), normal.y(),
            t.z(), b.z(), normal.z();
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    float h = payload.texture->height, w = payload.texture->width, u = payload.tex_coords.x(), v = payload.tex_coords.y();
    float dU = kh * kn * (payload.texture->getColorBilinear(u + 1 / w, v).norm() - payload.texture->getColorBilinear(u,
                                                                                                                     v).norm()); // note: it should be .norm(), we only need the value
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    float dV = kh * kn * (payload.texture->getColorBilinear(u, v + 1 / h).norm() -
                          payload.texture->getColorBilinear(u, v).norm());
    // Vector ln = (-dU, -dV, 1)
    Eigen::Vector3f ln(-dU, -dV, 1);
    // Position p = p + kn * n * h(u,v)
    point += kn * normal * payload.texture->getColorBilinear(u, v).norm();
    // Normal n = normalize(TBN * ln)
    normal = (TBN * ln).normalized();  // 法线贴图的作用，就是存一个相对值，可以用于更新point + normal

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto &light : lights) {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        Eigen::Vector3f l = light.position - point; // view_position -> light source
        Eigen::Vector3f v = eye_pos - point; // view_position -> eye

        float r2 = l.dot(l); // l cannot be normalized before, because r2 is the distance

        // note: l, v should be normalized
        l = l.normalized();
        v = v.normalized();

        // note: ka, kd, ks are scalars in lecture, but they are vectors here, so you should use cwiseProduct
        // ambient La=ka*Ia
        Eigen::Vector3f La = ka.cwiseProduct(amb_light_intensity); // cwiseProduct--dot product

        // diffuse Ld=kd(I/r^2)max(0, nl)
        Eigen::Vector3f Ld = kd.cwiseProduct(light.intensity / r2) * std::max(0.0f, normal.normalized().dot(l));

        // specular Ls=ks(I/r^2)max(0, nh)^p
        Eigen::Vector3f h = (l + v).normalized();
        Eigen::Vector3f Ls =
                ks.cwiseProduct(light.intensity / r2) * std::pow(std::max(0.0f, normal.normalized().dot(h)), p);

        result_color += (La + Ld + Ls);

    }

    return result_color * 255.f;
}


Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload &payload) {

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;  // 这个直接是在光栅化准备的时候直接指定的，没有读取材质的信息
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20,  20,  20},
                    {500, 500, 500}};
    auto l2 = light{{-20, 20,  0},
                    {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;


    float kh = 0.2, kn = 0.1;

    // TODO: Implement bump mapping here

    // Let n = normal = (x, y, z)
    auto x = normal.x(), y = normal.y(), z = normal.z();
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z)) 这一步是根据法线求切线
    Eigen::Vector3f t(x * y / std::sqrt(x * x + z * z), std::sqrt(x * x + z * z), z * y / sqrt(x * x + z * z));
    // Vector b = n cross product t 这一步是求binormal，即副切线
    Eigen::Vector3f b = normal.cross(t);
    // Matrix TBN = [t b n]
    Eigen::Matrix3f TBN;
    // 切线空间转视角空间
    TBN << t.x(), b.x(), normal.x(),
            t.y(), b.y(), normal.y(),
            t.z(), b.z(), normal.z();
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // line 343 - 352，这一段代码没看懂是怎么转换的
    float h = payload.texture->height, w = payload.texture->width, u = payload.tex_coords.x(), v = payload.tex_coords.y();
    float dU = kh * kn * (payload.texture->getColorBilinear(u + 1 / w, v).norm() - payload.texture->getColorBilinear(u,
                                                                                                                     v).norm()); // note: it should be .norm(), we only need the value
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    float dV = kh * kn * (payload.texture->getColorBilinear(u, v + 1 / h).norm() -
                          payload.texture->getColorBilinear(u, v).norm());
    // Vector ln = (-dU, -dV, 1)
    Eigen::Vector3f ln(-dU, -dV, 1);
    // Normal n = normalize(TBN * ln)
    normal = (TBN * ln).normalized();  // 这一步把tangent空间转到了view空间

    Eigen::Vector3f result_color = {0, 0, 0};
    result_color = normal;

    return result_color * 255.f;
}

// 在光栅器的层次上一共设置了六个参数
// Texture、vs、fs、mvp Matrix
int main(int argc, const char **argv) {
    std::vector<Triangle *> TriangleList;

//    float angle = 140.0;
    float angle = 140;
    bool command_line = false;

    std::string filename = "output.png";
    objl::Loader Loader;
    // Load .obj File
    bool loadout = Loader.LoadFile("../models/spot/spot_triangulated_good.obj");

    // decoder the mesh into triangleList, 将所有的信息转化成triangle结构统一存储
    for (auto mesh:Loader.LoadedMeshes) {
        for (int i = 0; i < mesh.Vertices.size(); i += 3) {
            Triangle *t = new Triangle();
            for (int j = 0; j < 3; j++) {
                t->setVertex(j, Vector4f(mesh.Vertices[i + j].Position.X, mesh.Vertices[i + j].Position.Y,
                                         mesh.Vertices[i + j].Position.Z, 1.0));
                t->setNormal(j, Vector3f(mesh.Vertices[i + j].Normal.X, mesh.Vertices[i + j].Normal.Y,
                                         mesh.Vertices[i + j].Normal.Z));
                t->setTexCoord(j, Vector2f(mesh.Vertices[i + j].TextureCoordinate.X,
                                           mesh.Vertices[i + j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }

    rst::rasterizer r(700, 700);

    auto texture_path = "../models/spot/hmap.jpg";
    r.set_texture(Texture(texture_path));

    // 最简单的理解方式，就是把std::function当成一个函数指针，重载的参数是<返回值(输入参数)>
    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = phong_fragment_shader;

    if (argc >= 2) {
        command_line = true;
        filename = std::string(argv[1]);

        if (argc == 3 && std::string(argv[2]) == "texture") {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
            texture_path = "../models/spot/spot_texture.png";
            r.set_texture(Texture(texture_path));
        } else if (argc == 3 && std::string(argv[2]) == "normal") {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        } else if (argc == 3 && std::string(argv[2]) == "phong") {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = phong_fragment_shader;
        } else if (argc == 3 && std::string(argv[2]) == "bump") {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        } else if (argc == 3 && std::string(argv[2]) == "displacement") {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = displacement_fragment_shader;
        }
    }

    Eigen::Vector3f eye_pos = {0, 0, 10};

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    int key = 0;
    int frame_count = 0;

//    auto model = get_model_matrix(angle);
//    auto view = get_view_matrix(eye_pos);
//    auto projection = get_projection_matrix(45.0, 1, 0.1, 50);
//
//    auto mv = view * model;
//    std::cout << mv << std::endl;
//    std::cout << std::endl;
//    auto mvp = projection * mv;
//    std::cout << mvp << std::endl;

    if (command_line) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        r.draw(TriangleList);
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
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        //r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        if (key == 'a') {
            angle -= 10;
        } else if (key == 'd') {
            angle += 10;
        }

    }
    return 0;
}