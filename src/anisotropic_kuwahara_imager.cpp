#include <vector>

#include <ai.h>
#include <omp.h>

#include "kuwahara.hpp"
#include "structure_tensor.hpp"


AI_IMAGER_NODE_EXPORT_METHODS(AnisotropicKuwaharaImagerMtd);


namespace
{
    static AtString radius_str("radius");
}

struct PixelData
{
    int idx;
    AtRGBA color;

    PixelData() : idx(0), color(AI_RGBA_ZERO) {}
    PixelData(int idx, AtRGBA color) : idx(idx), color(color) {}
};

struct AOVData
{
    const void*             bucket_data;
    int                     type;
    std::vector<PixelData>  pixels_data;

    AOVData(const void* bucket_data, int type) 
        : bucket_data(bucket_data), type(type) {}
};

node_parameters
{
    AiParameterInt("radius", 5);
}

node_initialize
{
}

node_update
{
}

imager_prepare
{
    // Set the imager schedule type to full frame always as 
    // we need access to all neighbouring pixels
    schedule = AtImagerSchedule::FULL_FRAME;
}

imager_evaluate
{
    // Node Parameters
    const int radius = AiNodeGetInt(node, radius_str);
    if (radius < 1)
        return;

    const float radiusf = static_cast<float>(radius);

    const float ellipse_min_radius = AiMax(.5f, AI_EPSILON);
    const int sectors_num = 8;  // N
    
    grid::GridSize grid(bucket_size_x, bucket_size_y);

    const int num_pixels = bucket_size_x * bucket_size_y;
    std::vector<AtRGBA> data(num_pixels, {AI_RGB_BLACK, 1.0f});

    // Iterator vars
    int aov_type = 0;
    const void *bucket_data;
    AtString output_name;

    while (AiOutputIteratorGetNext(iterator, &output_name, &aov_type, &bucket_data))
    {
        AtRGBA* rgba = (AtRGBA*)bucket_data;

        const cv::Mat structureTensor = structure_tensor::computeStructureTensor(rgba, grid, 3, 1.0f);

        // Iterate over the bucket
        #pragma omp parallel for
        for (int y = 0; y < bucket_size_y; ++y)
        {
            const int base_idx = y * bucket_size_x;

            for (int x = 0; x < bucket_size_x; ++x)
            {

                // Index of the pixel in the bucket
                const int idx = base_idx + x;

                const cv::Point2f center_point(static_cast<float>(x), static_cast<float>(y));

                // Compute local orientation and anisotropy
                // The local orientation is in radians
                const auto [orientation_rad, anisotropy] = 
                    structure_tensor::computeLocalOrientationAndAnisotropyAtPoint(structureTensor, x, y);

                // Compute the kernel shape
                float ellipse_major_radius = ((ellipse_min_radius + anisotropy) / ellipse_min_radius) * radiusf;  // a
                ellipse_major_radius = AiMax(radiusf, AiMin(ellipse_major_radius, 2.f * radiusf));  // r <= a <= 2r
            
                float ellipse_minor_radius = (ellipse_min_radius / (ellipse_min_radius + anisotropy)) * radiusf;  // b
                ellipse_minor_radius = AiMax(radiusf * .5f, AiMin(ellipse_minor_radius, radiusf));

                // Compute the scale matrix
                cv::Matx22f scale_mat(1.0f / ellipse_major_radius, .0f, .0f, 1.0f / ellipse_minor_radius);  // S = [1/a 0; 0 1/b]

                // Compute the rotation matrix
                const float orientation_cos = std::cos(orientation_rad);
                const float orientation_sin = std::sin(orientation_rad);
                cv::Matx22f rot_mat(orientation_cos, orientation_sin, -orientation_sin, orientation_cos);  // R = [cos(theta) sin(theta); -sin(theta) cos(theta)]

                // Compute the inverse affine matrix
                cv::Matx22f affine_mat = scale_mat * rot_mat ;  // M = S * R
            
                // Compute the bounding box of the ellipse
                const float half_width = 
                    std::ceil(std::abs(ellipse_major_radius * orientation_cos) + 
                    std::abs(ellipse_minor_radius * orientation_sin));

                const float half_height =
                    std::ceil(std::abs(ellipse_major_radius * orientation_sin) +
                    std::abs(ellipse_minor_radius * orientation_cos));

                // Définir le patch en fonction de la boîte englobante
                const int patch_min_x = AiMax(x - static_cast<int>(half_width),  0);
                const int patch_max_x = AiMin(x + static_cast<int>(half_width),  bucket_size_x);
                const int patch_min_y = AiMax(y - static_cast<int>(half_height), 0);
                const int patch_max_y = AiMin(y + static_cast<int>(half_height), bucket_size_y);

                std::array<float,  sectors_num> sum_weights           = {}; sum_weights.fill(0.0f);
                std::array<AtRGBA, sectors_num> sum_colors            = {}; sum_colors.fill(AI_RGB_BLACK);
                std::array<float,  sectors_num> sum_intensity         = {}; sum_intensity.fill(0.0f);
                std::array<float,  sectors_num> sum_intensity_squared = {}; sum_intensity_squared.fill(0.0f);

                // Iterate over the pixels in the patch
                for (int py = patch_min_y; py < patch_max_y; ++py)
                {
                    const int base_patch_idx = py * bucket_size_x;

                    for (int px = patch_min_x; px < patch_max_x; ++px)
                    {
                        const int patch_idx = base_patch_idx + px;

                        // Compute relative coordinates
                        const cv::Point2f patch_point(static_cast<float>(px), static_cast<float>(py));
                        const cv::Point2f local_point = patch_point - center_point;

                        // (u, v) = M * local_point
                        const cv::Point2f disc_point = affine_mat * local_point;
                        const float &u = disc_point.x;
                        const float &v = disc_point.y;
                        
                        // Check if the point is inside the ellipse
                        const bool is_inside = (u*u + v*v) <= 1.0f;
                        if (!is_inside)
                            continue;

                        // Get the pixel color and intensity
                        const AtRGBA& pixel_color = rgba[patch_idx];
                        const float I = (0.2126f * pixel_color.r) + (0.7152f * pixel_color.g) + (0.0722f * pixel_color.b);

                        // Compute the sector weights
                        std::array<float, sectors_num> sector_weights = {}; sector_weights.fill(0.0f);
                        float sum_weight = 0.0f;
                        
                        for (int i = 0; i < sectors_num; ++i)
                        {
                            float weight = anisotropic_kuwahara::computeSectorWeight(i, u, v, .4f, .4f, sectors_num);

                            sector_weights[i] = weight;
                            sum_weight += weight;
                        }

                        if (sum_weight <= AI_EPSILON)
                            continue;

                        for (int i = 0; i < sectors_num; ++i)
                        {
                            sector_weights[i] /= sum_weight;  // Normalize

                            sum_weights[i]    += sector_weights[i];
                            sum_colors[i]     += pixel_color * sector_weights[i];
                            sum_intensity[i]   += I * sector_weights[i];
                            sum_intensity_squared[i] += (I*I) * sector_weights[i];
                        }
                    }
                }

                AtRGBA color_total = AI_RGB_BLACK;
                float alpha_total = 0.0f;

                float k = 10.0f;
                int q = 8;

                for (int i = 0; i < sectors_num; ++i)
                {
                    if (sum_weights[i] <= AI_EPSILON) 
                        continue;

                    AtRGBA meanColor_i = sum_colors[i] / sum_weights[i];
                    float meanI = sum_intensity[i] / sum_weights[i];
                    float meanI2= sum_intensity_squared[i] / sum_weights[i];
                    float var_i = meanI2 - meanI*meanI;

                    float alpha_i = 1.0f / (1.0f + k * std::pow(var_i, (float)q));
                    color_total += meanColor_i * alpha_i;
                    alpha_total += alpha_i;
                }

                AtRGBA final_color = color_total / alpha_total;
                data[idx] = final_color;
            }
        }

        #pragma omp parallel for
        for (int y = 0; y < bucket_size_y; ++y)
        {
            const int base_idx = y * bucket_size_x;
    
            for (int x = 0; x < bucket_size_x; ++x)
            {
                const int idx = base_idx + x;
                rgba[idx] = data[idx];
            }
        }
    }
}

node_finish
{
}
