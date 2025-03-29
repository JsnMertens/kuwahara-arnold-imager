#include <array>
#include <cmath>
#include <vector>

#include <ai.h>
#include <omp.h>
#include <opencv2/opencv.hpp>

#include "kuwahara.hpp"
#include "structure_tensor.hpp"


AI_IMAGER_NODE_EXPORT_METHODS(AnisotropicKuwaharaImagerMtd);


namespace
{
    static AtString param_radius("radius");
    static AtString param_eccentricity("eccentricity");
    static AtString param_sharpness("sharpness");
    static AtString param_tensor_size("tensor_size");
    static AtString str_rgba("RGBA");
}  // anonymous namespace

node_parameters
{
    AiParameterInt(param_radius, 5);
    AiParameterFlt(param_eccentricity, 1.0f);
    AiParameterFlt(param_sharpness, 1.0f);
    AiParameterUInt(param_tensor_size, 3);
}

node_initialize
{
}

node_update
{
}

imager_prepare
{
    // Set the imager schedule type to full frame always as we need access to all neighbouring pixels
    schedule = AtImagerSchedule::FULL_FRAME;
}

imager_evaluate
{
    // Size of the filter kernel: 2xr + 1
    const int radius = AiNodeGetInt(node, param_radius);
    if (radius < 1)
    {
        AiMsgWarning("[Anisotropic Kuwahara] Radius must be greater than 0, Imager ignored.");
        return;
    }
    // Used for computations
    const float radiusf = static_cast<float>(radius);
    // Divide to invert the eccentricity.
    // Lower values create more isotropic kernels. eccentricity = 0.0f is isotropic
    float eccentricity = 1.0f / AiMin(1.0f, AiMax(AiNodeGetFlt(node, param_eccentricity), AI_EPSILON));
    // Sharpness over 3.0f create too much corrupted values
    const float sharpness = AiMin(3.0f, AiMax(AiNodeGetFlt(node, param_sharpness), AI_EPSILON));  
    // Sigma and size of the structure tensor, higher values create more "blurry" kernels
    const int tensor_size = AiNodeGetUInt(node, param_tensor_size);

    cv::Point2i img_size(bucket_size_x, bucket_size_y);
    // output kernel buffer
    const int num_pixels = bucket_size_x * bucket_size_y;
    std::vector<AtRGBA> kernel_buffer(num_pixels, AI_RGB_BLACK);

    // Output AOV name, set within AiOutputIteratorGetNext
    AtString output_name;
    // Ray type, e.g. AI_TYPE_RGB, set within AiOutputIteratorGetNext
    int aov_type = 0;
    // Pointer to the output buffer, set within AiOutputIteratorGetNext
    const void* bucket_data;

    // Iterate over the outputs (AOVs) 
    while (AiOutputIteratorGetNext(iterator, &output_name, &aov_type, &bucket_data))
    {
        if (output_name != str_rgba)
            continue;

        // Output color pointer
        // It's used to retrieve the color of the pixe and assign the new color to the pixel
        AtRGBA* output_ptr = nullptr;
        if (aov_type == AI_TYPE_RGBA)
            output_ptr = (AtRGBA*)bucket_data;
        else
            output_ptr = (AtRGBA*)bucket_data;  // TODO: Not implemented yet for other types
            
        // Compute the structure tensor for the current bucket,
        // it will be used to get the local orientation and the anisotropy
        const cv::Mat tensor =
            structure_tensor::computeStructureTensor(output_ptr, img_size, tensor_size);

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
                auto kernel_transform_mat = anisotropic_kuwahara::computeEllipseToUnitDiskMatrix(
                    ellipse_radius, orientation_cos, orientation_sin
                );

                // Compute the elliptical kernel bounding box
                auto kernel_bbox = anisotropic_kuwahara::computeEllipticalKernelBbox(
                    x, y, ellipse_radius, img_size, orientation_cos, orientation_sin
                );

                // Allocate output kernel data
                std::array<float,  anisotropic_kuwahara::sector_size> sum_weights = {};
                std::array<AtRGBA, anisotropic_kuwahara::sector_size> sum_colors = {};
                std::array<AtRGBA, anisotropic_kuwahara::sector_size> sum_colors_squared = {};
                sum_weights.fill(0.0f);
                sum_colors.fill(AI_RGB_BLACK);
                sum_colors_squared.fill(AI_RGB_BLACK);

                // Iterate over the pixels in the kernel (patch)
                for (int sub_y = kernel_bbox.min.y; sub_y < kernel_bbox.max.y; ++sub_y)
                {
                    const int base_sub_idx = sub_y * bucket_size_x;
                    for (int sub_x = kernel_bbox.min.x; sub_x < kernel_bbox.max.x; ++sub_x)
                    {
                        // Sub pixel index, inside the kernel BBox
                        const int sub_idx = base_sub_idx + sub_x;

                        // Compute relative coordinates
                        const cv::Point2f patch_point(static_cast<float>(sub_x), static_cast<float>(sub_y));
                        const cv::Point2f local_point = patch_point - center_point;

                        // (u, v) = M * local_point
                        cv::Point2f disk_point = kernel_transform_mat * local_point;
                        const float& u = disk_point.x;
                        const float& v = disk_point.y;
                        
                        // Check if the point is inside the elliptical kernel, if not skip
                        const bool is_inside = (u*u + v*v) <= 1.0f;
                        if (!is_inside)
                            continue;

                        // Compute the sector weights
                        std::array<float, anisotropic_kuwahara::sector_size> sector_weights = {};
                        sector_weights.fill(0.0f);
                        auto sum_weight = anisotropic_kuwahara::computeSectorWeight(u, v, radiusf, sector_weights);
                        if (sum_weight <= AI_EPSILON)
                            continue;
                        
                        // Compute a Gaussian weight so that pixels further from the kernel origin have less weight
                        float radial_gaussian_weight = std::exp(-AI_PI * disk_point.dot(disk_point)) / sum_weight;

                        // Get the pixel color
                        const AtRGBA& pixel_color = output_ptr[sub_idx];

                        // Smooth the weight with a Gaussian and 
                        // weights the local average by the sector weights 
                        for (int i = 0; i < anisotropic_kuwahara::sector_size; ++i)
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
                
                for (int i = 0; i < anisotropic_kuwahara::sector_size; ++i)
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

                // Keep the original color if color is corrupted
                const AtRGBA final_color = final_color_sum / standard_deviation_sum;
                if (AiColorIsSmall(AtRGB(final_color)) || !AiRGBAIsFinite(final_color))
                    kernel_buffer[idx] = output_ptr[idx];
                else
                    kernel_buffer[idx] = final_color;

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
                output_ptr[idx] = kernel_buffer[idx];
            }
        }

    }  // while
}  // imager_evaluate

node_finish
{
}
