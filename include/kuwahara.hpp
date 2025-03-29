#ifndef KUWAHARA_ARNOLD_KUWAHARA_H_
#define KUWAHARA_ARNOLD_KUWAHARA_H_

#include <array>
#include <cmath>

#include <ai.h>
#include <opencv2/opencv.hpp>

#include "grid.hpp"

static constexpr float SQRT2_OVER2 = 0.70710678f;

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
    const float u,
    const float v,
    const float radius,
    std::array<float, 8>& sector_weights
)
{
    // Output
    float sum_weight = 0.0f;

    // Parameters from the Paper:
    // "Anisotropic Kuwahara Filtering with Polynomial Weighting Functions"
    static const float zeta = 2.0f / radius;
    static const float gamma = AI_PI / 8.0f; // 3π/8
    static const float eta = zeta + std::cos(gamma) / AiSqr(std::sin(gamma));

    // Compute polynomial weights for each even sector
    float poly_u = zeta - eta * u * u;
    float poly_v = zeta - eta * v * v;

    sector_weights[0] = AiSqr(AiMax(0.0f,  v + poly_u));
    sector_weights[2] = AiSqr(AiMax(0.0f, -u + poly_v));
    sector_weights[4] = AiSqr(AiMax(0.0f, -v + poly_u));
    sector_weights[6] = AiSqr(AiMax(0.0f,  u + poly_v));

    // Rotate by 45 degrees and the compute polynomial weights for each odd sector
    const float rotated_u = SQRT2_OVER2 * (u - v);
    const float rotated_v = SQRT2_OVER2 * (u + v);

    poly_u = zeta - eta * rotated_u * rotated_u;
    poly_v = zeta - eta * rotated_v * rotated_v;

    sector_weights[1] = AiSqr(AiMax(0.0f,  rotated_v + poly_u));
    sector_weights[3] = AiSqr(AiMax(0.0f, -rotated_u + poly_v));
    sector_weights[5] = AiSqr(AiMax(0.0f, -rotated_v + poly_u));
    sector_weights[7] = AiSqr(AiMax(0.0f,  rotated_u + poly_v));

    // sum weights
    sum_weight += sector_weights[0];
    sum_weight += sector_weights[1];
    sum_weight += sector_weights[2];
    sum_weight += sector_weights[3];
    sum_weight += sector_weights[4];
    sum_weight += sector_weights[5];
    sum_weight += sector_weights[6];
    sum_weight += sector_weights[7];

    return sum_weight;
}


} // namespace anisotropic_kuwahara

#endif // KUWAHARA_ARNOLD_KUWAHARA_H_
