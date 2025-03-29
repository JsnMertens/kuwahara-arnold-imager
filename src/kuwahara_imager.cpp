#include <vector>

#include <ai.h>
#include <omp.h>
#include <opencv2/opencv.hpp>

#include "kuwahara.hpp"


AI_IMAGER_NODE_EXPORT_METHODS(KuwaharaImagerMtd);


namespace
{
    static AtString param_radius("radius");
    static AtString str_rgba("RGBA");
}

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
    // Set the imager schedule type to full frame always as we need access to all neighbouring pixels
    schedule = AtImagerSchedule::FULL_FRAME;
}

imager_evaluate
{
    // Size of the filter kernel: 2xr + 1
    const int radius = AiNodeGetInt(node, param_radius);
    if (radius < 1)
    {
        AiMsgWarning("[Kuwahara] Radius must be greater than 0, Imager ignored.");
        return;
    }

    cv::Point2i img_size(bucket_size_x, bucket_size_y);
    // Output kernel buffer
    const int num_pixels = bucket_size_x * bucket_size_y;
    std::vector<AtRGBA> kernel_buffer(num_pixels, {AI_RGB_BLACK, 1.0f});

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
        AtRGBA* output_ptr = (AtRGBA*)bucket_data;

        // Iterate over the pixels
        #pragma omp parallel for
        for (int y = 0; y < bucket_size_y; ++y)
        {
            const int base_idx = y * bucket_size_x;
            for (int x = 0; x < bucket_size_x; ++x)
            {
                // Index of the pixel in the bucket
                const int idx = base_idx + x;
                // Center point of the kernel, before to compute the quadrants
                const cv::Point2i center_point(x, y);

                AtRGBA best_color = output_ptr[idx]; 
                float best_variance = AI_BIG;

                // Iterate over the 4 quadrants
                for (auto quadrant : kuwahara::quadrants)
                {
                    auto kernel_bbox = kuwahara::computeQuadrantKernelBbox(center_point, radius, img_size, quadrant);
                    const auto [mean_color, variance] = kuwahara::computeRegion(output_ptr, img_size, kernel_bbox);

                    if (variance > best_variance)
                        continue;
                    
                    best_variance = variance;
                    best_color = mean_color;
                }
                kernel_buffer[idx] = best_color;
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
