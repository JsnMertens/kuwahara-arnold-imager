#ifndef KUWAHARA_ARNOLD_KUWAHARA_H_
#define KUWAHARA_ARNOLD_KUWAHARA_H_

#include <array>
#include <cmath>

#include <ai.h>
#include <opencv2/opencv.hpp>

#include "grid.hpp"


namespace kuwahara
{
    
using namespace grid;

enum class Quadrant
{
    kTopLeft,
    kTopRight,
    kBottomLeft,
    kBottomRight
};

constexpr std::array<Quadrant, 4> quadrants = {
    Quadrant::kTopLeft,
    Quadrant::kTopRight,
    Quadrant::kBottomLeft,
    Quadrant::kBottomRight
};

inline
GridRegion computeQuadrantRegion(const GridPoint& center, const GridSize& grid, int radius, Quadrant quadrant)
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
void computeRegion(AtRGBA* color, GridSize grid, const GridRegion& region, AtRGBA& mean, float& variance)
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

} // namespace kuwahara

namespace anisotropic_kuwahara
{

inline
std::pair<float, float> computePolynomialEllipticalKernelShape (
    const float anisotropy,
    const float radius,
    const float ellipse_min_radius = 1.0f  // alpha
)
{
    float ellipse_major_radius = ((ellipse_min_radius + anisotropy) / ellipse_min_radius) * radius;  // a
    ellipse_major_radius = AiMax(radius, AiMin(ellipse_major_radius, 2.f * radius));  // r <= a <= 2r

    float ellipse_minor_radius = (ellipse_min_radius / (ellipse_min_radius + anisotropy)) * radius;  // b
    ellipse_minor_radius = AiMax(radius * .5f, AiMin(ellipse_minor_radius, radius));  // r/2 <= b <= r

    return std::make_pair(ellipse_major_radius, ellipse_minor_radius);    
}

inline
float computeSectorWeight(
    const int sector_idx,
    const float u,
    const float v,
    const float sigma_angle = .4f,
    const float sigma_rad = .4f,
    const int sector_num = 8
)
{
    // Compute polar coordinates 
    float r = std::sqrtf(u*u + v*v);
    float phi = std::atan2(v, u);
    if (phi < 0)
        phi += AI_PITIMES2;  // if angle is negative, add 2π to make it positive

    // Compute the center of the sector
    float sector_center = (sector_idx + 0.5f) * (AI_PITIMES2 / sector_num);

    // Angular difference (make sure it is in [0, π])
    float delta_phi = std::fabs(phi - sector_center);
    if (delta_phi > AI_PI)
        delta_phi = AI_PITIMES2 - delta_phi;

    float half_sector = AI_PI / sector_num;
    if (delta_phi > half_sector)
        return 0.0f;  // Outside the sector

    // Angular gaussian
    float angular_weight = std::exp(- (delta_phi * delta_phi) / (2.0f * (sigma_angle * sigma_angle)));

    // Radial gaussian
    float radial_weight = std::exp(- (r * r) / (2.0f * (sigma_rad * sigma_rad)));

    return angular_weight * radial_weight;
}


inline
float computeGaussianWeight(
    const float x,
    const float y,
    const float sigma = 1.0f
)
{
    return std::exp(- (x * x + y * y) / (2.0f * (sigma * sigma)));
}

} // namespace anisotropic_kuwahara

#endif // KUWAHARA_ARNOLD_KUWAHARA_H_
