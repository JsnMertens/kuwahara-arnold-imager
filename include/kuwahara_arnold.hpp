#ifndef KUWAHARA_ARNOLD_KUWAHARA_H_
#define KUWAHARA_ARNOLD_KUWAHARA_H_

#include <array>

#include <ai.h>
#include <opencv2/opencv.hpp>

#include "grid.hpp"


namespace kuwahara_arnold
{
    
using namespace grid;

enum class Quadrant
{
    kTopLeft,
    kTopRight,
    kBottomLeft,
    kBottomRight
};

constexpr std::array<kuwahara_arnold::Quadrant, 4> quadrants = {
    kuwahara_arnold::Quadrant::kTopLeft,
    kuwahara_arnold::Quadrant::kTopRight,
    kuwahara_arnold::Quadrant::kBottomLeft,
    kuwahara_arnold::Quadrant::kBottomRight
};

inline
GridRegion ComputeQuadrantRegion(const GridPoint& center, const GridSize& grid, int radius, Quadrant quadrant)
{
    GridRegion region;

    switch (quadrant)
    {
        case Quadrant::kTopLeft:
            region.min.x = AiMax(0, center.x - radius);
            region.max.x = center.x;
            region.min.y = AiMax(0, center.y - radius);
            region.max.y = center.y;
            break;

        case Quadrant::kTopRight:
            region.min.x = center.x;
            region.max.x = AiMin(grid.x, center.x + radius);
            region.min.y = AiMax(0, center.y - radius);
            region.max.y = center.y;
            break;

        case Quadrant::kBottomLeft:
            region.min.x = AiMax(0, center.x - radius);
            region.max.x = center.x;
            region.min.y = center.y;
            region.max.y = AiMin(grid.y, center.y + radius);
            break;

        case Quadrant::kBottomRight:
            region.min.x = center.x;
            region.max.x = AiMin(grid.x, center.x + radius);
            region.min.y = center.y;
            region.max.y = AiMin(grid.y, center.y + radius);
            break;
    }

    return region;
}

inline
void ComputeRegion(AtRGBA* color, GridSize grid, const GridRegion& region, AtRGBA& mean, float& variance)
{
    int count = 0;
    mean = AI_RGBA_ZERO;

    for (int y = region.min.y; y < region.max.y; ++y)
    {
        for (int x = region.min.x; x < region.max.x; ++x)
        {
            int idx = y * grid.x + x;

            mean += color[idx];
            count++;
        }
    }
    if (count == 0)
    {
        mean = AI_RGBA_ZERO;
        variance = AI_BIG;
        return;
    }
    float inv_countf = 1.0f / static_cast<float>(count);
    mean *= inv_countf;

    float sum_var = 0.0f;
    for (int y = region.min.y; y < region.max.y; ++y)
    {
        for (int x = region.min.x; x < region.max.x; ++x)
        {
            int idx = y * grid.x + x;
            float dr = color[idx].r - mean.r;
            float dg = color[idx].g - mean.g;
            float db = color[idx].b - mean.b;
            sum_var += (dr * dr + dg * dg + db * db);
        }
    }
    variance = sum_var / count;
}

} // namespace kuwahara_arnold

namespace anisotropic_kuwahara_arnold
{



} // namespace anisotropic_kuwahara_arnold

#endif // KUWAHARA_ARNOLD_KUWAHARA_H_
