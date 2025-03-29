#include <array>
#include <cmath>
#include <vector>

#include <ai.h>
#include <omp.h>

#include "kuwahara.hpp"
#include "structure_tensor.hpp"


AI_IMAGER_NODE_EXPORT_METHODS(AnisotropicKuwaharaImagerMtd);

namespace
{
    static AtString param_radius("radius");
    static AtString param_eccentricity("eccentricity");
    static AtString param_sharpness("sharpness");

    static AtString param_tensor_size("tensor_size");
    static AtString param_tensor_sigma("tensor_sigma");
}  // anonymous namespace

node_parameters
{
    AiParameterInt(param_radius, 5);
    AiParameterFlt(param_eccentricity, 1.0f);
    AiParameterFlt(param_sharpness, 1.0f);
    
    AiParameterInt("tensor_size", 3);
    AiParameterFlt("tensor_sigma", 1.0f);
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
    const int radius = AiNodeGetInt(node, param_radius);
    if (radius < 1)
        return;

    // Used for computations
    const float radiusf = static_cast<float>(radius);
    // Divide to invert the eccentricity.
    // Lower values create more isotropic kernels. eccentricity = 0.0f is isotropic
    float eccentricity = 1.0f / AiMin(1.0f, AiMax(AiNodeGetFlt(node, param_eccentricity), AI_EPSILON));
    // Sharpness over 3.0f create too much corrupted values
    const float sharpness = AiMin(3.0f, AiMax(AiNodeGetFlt(node, param_sharpness), AI_EPSILON));  
    // Could be 4 or 8, but 8 is better for anisotropic kernels and the code is designed for 8
    const int num_sectors = 8;  // N
    // data is the output kernel buffer
    const int num_pixels = bucket_size_x * bucket_size_y;
    std::vector<AtRGBA> data(num_pixels, AI_RGB_BLACK);

    const int tensor_size = AiNodeGetInt(node, param_tensor_size);
    const float tensor_sigma = AiNodeGetFlt(node, param_tensor_sigma);

    // Pointer to the output buffer, set within AiOutputIteratorGetNext
    const void* bucket_data;
    // Ray type, e.g. AI_TYPE_RGB, set within AiOutputIteratorGetNext
    int aov_type = 0;
    // Output AOV name, set within AiOutputIteratorGetNext
    AtString output_name;

    // Iterate over the outputs (AOVs) 
    while (AiOutputIteratorGetNext(iterator, &output_name, &aov_type, &bucket_data))
    {
        AtRGBA* output_pixel_ptr = nullptr;
        if (aov_type == AI_TYPE_RGBA)
            output_pixel_ptr = (AtRGBA*)bucket_data;
        else
            output_pixel_ptr = (AtRGBA*)bucket_data;  // TODO: Not implemented yet for other types
            
        // Compute the structure tensor for the current bucket, it will be used to get the local orientation and the anisotropy
        const cv::Mat tensor =
            structure_tensor::computeStructureTensor(output_pixel_ptr, bucket_size_x, bucket_size_y, tensor_size, tensor_sigma);

        // Iterate over the pixels
        #pragma omp parallel for
        for (int y = 0; y < bucket_size_y; ++y)
        {
            const int base_idx = y * bucket_size_x;

            for (int x = 0; x < bucket_size_x; ++x)
            {
                // Index of the pixel in the bucket
                const int idx = base_idx + x;

                // Center point of the kernel
                const cv::Point2f center_point(static_cast<float>(x), static_cast<float>(y));

                // Compute local orientation and anisotropy
                // The local orientation is in radians
                const auto [orientation_rad, anisotropy] = 
                    structure_tensor::computeLocalOrientationAndAnisotropyAtPoint(tensor, x, y);

                // Compute the kernel shape; a anb b
                auto ellipse_radius =
                    anisotropic_kuwahara::computePolynomialEllipticalKernelShape(anisotropy, radiusf, eccentricity);

                // Compute the Kernel Transform Matrix, 
                // This matrix transforms the coordinates from the ellipse to a circular kernel
                const float orientation_cos = std::cos(orientation_rad);
                const float orientation_sin = std::sin(orientation_rad);

                cv::Matx22f kernel_transform_mat = anisotropic_kuwahara::computeEllipseToUnitDiskMatrix(
                    ellipse_radius, orientation_cos, orientation_sin
                );

                // Compute the elliptical kernel bounding box
                anisotropic_kuwahara::KernelBbox kernel_bbox;
                anisotropic_kuwahara::computeEllipticalKernelBbox(
                    x, y,
                    ellipse_radius,
                    orientation_cos, orientation_sin,
                    bucket_size_x, bucket_size_y,
                    kernel_bbox
                );

                const int& kernel_min_x = kernel_bbox[0][0];
                const int& kernel_max_x = kernel_bbox[0][1];
                const int& kernel_min_y = kernel_bbox[1][0];
                const int& kernel_max_y = kernel_bbox[1][1];

                // Allocate output kernel data
                std::array<float,  num_sectors> sum_weights        = {}; sum_weights.fill(0.0f);
                std::array<AtRGBA, num_sectors> sum_colors         = {}; sum_colors.fill(AI_RGB_BLACK);
                std::array<AtRGBA, num_sectors> sum_colors_squared = {}; sum_colors_squared.fill(AI_RGB_BLACK);

                // Iterate over the pixels in the kernel (patch)
                for (int sub_y = kernel_min_y; sub_y < kernel_max_y; ++sub_y)
                {
                    const int base_sub_idx = sub_y * bucket_size_x;

                    for (int sub_x = kernel_min_x; sub_x < kernel_max_x; ++sub_x)
                    {
                        const int sub_idx = base_sub_idx + sub_x;

                        // Compute relative coordinates
                        const cv::Point2f patch_point(static_cast<float>(sub_x), static_cast<float>(sub_y));
                        const cv::Point2f local_point = patch_point - center_point;

                        // (u, v) = M * local_point
                        cv::Point2f disk_point = kernel_transform_mat * local_point;
                        const float& u = disk_point.x;
                        const float& v = disk_point.y;
                        
                        // Check if the point is inside the ellipse, if not skip
                        const bool is_inside = (u*u + v*v) <= 1.0f;
                        if (!is_inside)
                            continue;

                        // Compute the sector weights
                        std::array<float, num_sectors> sector_weights = {};
                        float sum_weight = anisotropic_kuwahara::computeSectorWeight(u, v, radiusf, sector_weights);

                        if (sum_weight <= AI_EPSILON)
                            continue;
                        
                        // Compute a Gaussian weight so that pixels further from the kernel origin have less weight
                        float radial_gaussian_weight = std::exp(-AI_PI * disk_point.dot(disk_point)) / sum_weight;

                        // Get the pixel color
                        const AtRGBA& pixel_color = output_pixel_ptr[sub_idx];

                        for (int i = 0; i < num_sectors; ++i)
                        {
                            sector_weights[i]     *= radial_gaussian_weight;

                            sum_weights[i]        += sector_weights[i];
                            sum_colors[i]         += pixel_color * sector_weights[i];
                            sum_colors_squared[i] += pixel_color * pixel_color * sector_weights[i];
                        }
                    }  // for sub_x
                }  // for sub_y

                // Compute the mean color and standard deviation for each sector
                // and then compute the final color as a weighted sum of the mean colors
                AtRGBA final_color_sum = AI_RGB_BLACK;
                float standard_deviation_sum = 0.0f;
                
                for (int i = 0; i < num_sectors; ++i)
                {
                    if (sum_weights[i] <= AI_EPSILON) 
                        continue;

                    AtRGBA mean_color = sum_colors[i] / sum_weights[i];
                    AtRGBA mean_color_squared = sum_colors_squared[i] / sum_weights[i];

                    float variance_r = std::fabs(mean_color_squared.r - mean_color.r * mean_color.r);
                    float variance_g = std::fabs(mean_color_squared.g - mean_color.g * mean_color.g);
                    float variance_b = std::fabs(mean_color_squared.b - mean_color.b * mean_color.b);
                    float variance = variance_r + variance_g + variance_b;

                    float standard_deviation = std::sqrt(variance);
                    float standard_deviation_factor =
                        1.0f / (1.0f * std::pow(standard_deviation, (8.0f * sharpness)));  // q = 8.0f
                    
                    final_color_sum += mean_color * standard_deviation_factor;
                    standard_deviation_sum += standard_deviation_factor;
                }

                const AtRGBA final_color = final_color_sum / standard_deviation_sum;
                if (AiColorIsSmall(AtRGB(final_color)) || !AiRGBAIsFinite(final_color))
                    data[idx] = output_pixel_ptr[idx];  // Keep the original color if color is corrupted
                else
                    data[idx] = (final_color_sum / standard_deviation_sum);

            }  // for x
        }  // for y

        // Assign the data back to the bucket
        #pragma omp parallel for
        for (int y = 0; y < bucket_size_y; ++y)
        {
            const int base_idx = y * bucket_size_x;
    
            for (int x = 0; x < bucket_size_x; ++x)
            {
                const int idx = base_idx + x;
                output_pixel_ptr[idx] = data[idx];
            }
        }

    }  // while
}  // imager_evaluate

node_finish
{
}
