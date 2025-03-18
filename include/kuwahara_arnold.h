#ifndef KUWAHARAARNOLD_KUWAHARA_H_
#define KUWAHARAARNOLD_KUWAHARA_H_

#include <ai.h>

#include "grid.h"

#include <iostream>

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
        mean /= count;

        float sum_var = 0.0f;
        for (int y = region.min.y; y < region.max.y; ++y)
        {
            for (int xx = region.min.x; xx < region.max.x; ++xx)
            {
                int idx = y * grid.x + xx;
                float dr = color[idx].r - mean.r;
                float dg = color[idx].g - mean.g;
                float db = color[idx].b - mean.b;
                sum_var += (dr * dr + dg * dg + db * db);
            }
        }
        variance = sum_var / count;
    }

} // namespace kuwahara_arnold

#endif // KUWAHARAARNOLD_KUWAHARA_H_
