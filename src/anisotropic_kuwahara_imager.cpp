#include <vector>

#include <ai.h>

#include "kuwahara_arnold.hpp"
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
    const void* bucket_data;
    int         type;
    std::vector<PixelData> pixels_data;

    AOVData(const void* bucket_data, int type) : bucket_data(bucket_data), type(type) {}
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
    // Set the imager schedule type to full frame always as we need access to all neighbouring pixels
    schedule = AtImagerSchedule::FULL_FRAME;
}

imager_evaluate
{
    // Node Parameters
    const int radius = AiNodeGetInt(node, radius_str);

    std::vector<AOVData> aovs;
    grid::GridSize       grid(bucket_size_x, bucket_size_y);

    // Init vars for Iterator
    int         aov_type = 0;
    const void  *bucket_data;
    AtString    output_name;

    while (AiOutputIteratorGetNext(iterator, &output_name, &aov_type, &bucket_data))
    {
        auto aov_data = AOVData(bucket_data, aov_type);
        AtRGBA* rgba = (AtRGBA*)bucket_data;

        // cv::Mat structure_tensor = structure_tensor::computeStructureTensor(rgba, bucket_size_x, bucket_size_y, 3, 1.0);
        // Mat mat(bucket_size_y, bucket_size_x, CV_32FC3);

        // Pour chaque pixel du bucket
        for (int y = 0; y < bucket_size_y; ++y)
        {
            for (int x = 0; x < bucket_size_x; ++x)
            {
                int idx = y * bucket_size_x + x;

                grid::GridPoint  center(x, y);

                AtRGBA  best_color      = rgba[idx]; // Valeur par défaut
                float   best_variance   = AI_BIG;

                AtRGBA  mean_color      = AI_RGBA_ZERO;
                float   variance        = 0.0f;

                // Iterate over the 4 quadrants
                for (auto quadrant : kuwahara_arnold::quadrants)
                {
                    grid::GridRegion quadrant_region = kuwahara_arnold::ComputeQuadrantRegion(center, grid, radius, quadrant);
                    kuwahara_arnold::ComputeRegion(rgba, grid, quadrant_region, mean_color, variance);

                    if (variance < best_variance)
                    {
                        best_variance = variance;
                        best_color = mean_color;
                    }
                }

                // Store pixel data for each index
                aov_data.pixels_data.push_back(PixelData(idx, best_color));
            }
        }
        aovs.push_back(aov_data);
    }

    // Set the output color for each AOV
    for (const auto &aov : aovs)
    {
        AtRGBA* output_color = (AtRGBA*)aov.bucket_data;
        for (const auto pixel_data : aov.pixels_data)
        {
            output_color[pixel_data.idx] = pixel_data.color;
        }
    }
}

node_finish
{
}
