#ifndef KUWAHARA_ARNOLD_KUWAHARA_H_
#define KUWAHARA_ARNOLD_KUWAHARA_H_

#include <array>
#include <cmath>

#include <ai.h>
#include <opencv2/opencv.hpp>

static constexpr float SQRT2_OVER2 = 0.70710678f;

namespace kuwahara
{

struct KernelBbox
{
    cv::Point2i min;
    cv::Point2i max;
};
    
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
KernelBbox computeQuadrantKernelBbox(
    const cv::Point2i& center,
    const int radius,
    const cv::Point2i& img_size,
    const Quadrant quadrant
)
{
    KernelBbox bbox;
    switch (quadrant)
    {
        case Quadrant::kTopLeft:
            bbox.min.x = AiMax(0, center.x - radius);
            bbox.max.x = center.x;
            bbox.min.y = AiMax(0, center.y - radius);
            bbox.max.y = center.y;
            break;

        case Quadrant::kTopRight:
            bbox.min.x = center.x;
            bbox.max.x = AiMin(img_size.x, center.x + radius);
            bbox.min.y = AiMax(0, center.y - radius);
            bbox.max.y = center.y;
            break;

        case Quadrant::kBottomLeft:
            bbox.min.x = AiMax(0, center.x - radius);
            bbox.max.x = center.x;
            bbox.min.y = center.y;
            bbox.max.y = AiMin(img_size.y, center.y + radius);
            break;

        case Quadrant::kBottomRight:
            bbox.min.x = center.x;
            bbox.max.x = AiMin(img_size.x, center.x + radius);
            bbox.min.y = center.y;
            bbox.max.y = AiMin(img_size.y, center.y + radius);
            break;
    }
    return bbox;
}

inline
std::pair<AtRGBA, float> computeRegion(
    const AtRGBA* color,
    const cv::Point2i& img_size,
    const KernelBbox& kernel_bbox)
{
    int count = 0;
    AtRGBA mean = AI_RGBA_ZERO;
    float variance = AI_BIG;

    // Compute mean color
    for (int y = kernel_bbox.min.y; y < kernel_bbox.max.y; ++y)
    {
        const int base_idx = y * img_size.x;
        for (int x = kernel_bbox.min.x; x < kernel_bbox.max.x; ++x)
        {
            const int idx = base_idx + x;
            mean += color[idx];
            count++;
        }
    }
    // If the kernel is empty, return zero
    if (count == 0)
        return std::make_pair(AI_RGBA_ZERO, AI_BIG);

    // Normalize mean color
    mean *= (1.0f / count);
    
    // Compute variance
    float sum_var = 0.0f;
    for (int y = kernel_bbox.min.y; y < kernel_bbox.max.y; ++y)
    {
        const int base_idx = y * img_size.x;
        for (int x = kernel_bbox.min.x; x < kernel_bbox.max.x; ++x)
        {
            const int idx = base_idx + x;

            const float dr = color[idx].r - mean.r;
            const float dg = color[idx].g - mean.g;
            const float db = color[idx].b - mean.b;
            
            sum_var += (dr*dr + dg*dg + db*db);
        }
    }
    // Normalize variance
    variance = sum_var / count;

    return std::make_pair(mean, variance);
}

} // namespace kuwahara

namespace anisotropic_kuwahara
{

// Number of sectors in the kernel.
// Could be 4 or 8, but 8 is better for anisotropic kernels and the code is designed for 8.
static constexpr int sector_size = 8;

using KernelBbox = kuwahara::KernelBbox;

struct EllipseRadius
{
    float max;
    float min;

    EllipseRadius(float max_radius, float min_radius)
        : max(max_radius), min(min_radius) {}
};

inline
EllipseRadius computePolynomialEllipticalKernelShape (
    const float anisotropy,
    const float radius,
    const float eccentricity = 1.0f  // alpha
)
{
    float ellipse_major_radius = ((eccentricity + anisotropy) / eccentricity) * radius;  // a
    ellipse_major_radius = AiMax(radius, AiMin(ellipse_major_radius, 2.0f * radius));  // r <= a <= 2r

    float ellipse_minor_radius = (eccentricity / (eccentricity + anisotropy)) * radius;  // b
    ellipse_minor_radius = AiMax(radius * 0.5f, AiMin(ellipse_minor_radius, radius));  // r/2 <= b <= r

    return EllipseRadius(ellipse_major_radius, ellipse_minor_radius);
}

inline
KernelBbox computeEllipticalKernelBbox(
    const int x,
    const int y,
    const EllipseRadius& ellipse_radius,
    const cv::Point2i& max_size,
    const float orientation_cos,
    const float orientation_sin
)
{
    const int half_width = static_cast<int>(
        std::ceil(std::abs(ellipse_radius.max * orientation_cos) + 
        std::abs(ellipse_radius.min * orientation_sin))
    );

    const int half_height = static_cast<int>(
        std::ceil(std::abs(ellipse_radius.max * orientation_sin) +
        std::abs(ellipse_radius.min * orientation_cos))
    );

    // Compute the kernel bounding box
    KernelBbox bbox;
    bbox.min.x = AiMax(x - half_width,  0);
    bbox.min.y = AiMax(y - half_height, 0);
    bbox.max.x = AiMin(x + half_width,  max_size.x);
    bbox.max.y = AiMin(y + half_height, max_size.y);

    return bbox;
}

inline
cv::Matx22f computeEllipseToUnitDiskMatrix(
    const EllipseRadius& ellipse_radius,
    float orientation_cos,
    float orientation_sin
)
{
    cv::Matx22f scale_mat(
        1.0f / ellipse_radius.max, 0.0f,
        0.0f, 1.0f / ellipse_radius.min
    );  // S = [1/a 0; 0 1/b]

    cv::Matx22f rot_mat(
         orientation_cos, orientation_sin,
        -orientation_sin, orientation_cos
    );  // R = [cos(theta) sin(theta); -sin(theta) cos(theta)]

    return scale_mat * rot_mat;  // M = S * R
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
    static const float gamma = AI_PI / 8.0f;
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
