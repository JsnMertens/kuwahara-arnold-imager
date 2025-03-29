#include <vector>

#include <ai.h>

#include "kuwahara.hpp"


AI_IMAGER_NODE_EXPORT_METHODS(ClassicKuwaharaImagerMtd);

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

namespace
{
    static AtString param_radius("radius");
}

imager_prepare
{
    // Set the imager schedule type to full frame always as we need access to all neighbouring pixels
    schedule = AtImagerSchedule::FULL_FRAME;
}

imager_evaluate
{
    // Node Parameters
    const int radius = AiNodeGetInt(node, param_radius);

    grid::GridSize grid(bucket_size_x, bucket_size_y);

    const int num_pixels = bucket_size_x * bucket_size_y;
    std::vector<AtRGBA> data(num_pixels, {AI_RGB_BLACK, 1.0f});

    // Init vars for Iterator
    int aov_type = 0;
    const void* bucket_data;
    AtString output_name;

    while (AiOutputIteratorGetNext(iterator, &output_name, &aov_type, &bucket_data))
    {
        AtRGBA* rgba = (AtRGBA*)bucket_data;

        #pragma omp parallel for
        for (int y = 0; y < bucket_size_y; ++y)
        {
            const int base_idx = y * bucket_size_x;

            for (int x = 0; x < bucket_size_x; ++x)
            {
                const int idx = base_idx + x;

                grid::GridPoint  center(x, y);

                AtRGBA  best_color      = rgba[idx]; // Valeur par défaut
                float   best_variance   = AI_BIG;

                AtRGBA  mean_color      = AI_RGBA_ZERO;
                float   variance        = 0.0f;

                // Iterate over the 4 quadrants
                for (auto quadrant : kuwahara::quadrants)
                {
                    grid::GridRegion quadrant_region = kuwahara::computeQuadrantRegion(center, grid, radius, quadrant);
                    kuwahara::computeRegion(rgba, grid, quadrant_region, mean_color, variance);

                    if (variance < best_variance)
                    {
                        best_variance = variance;
                        best_color = mean_color;
                    }
                }
                
                data[idx] = best_color;
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
