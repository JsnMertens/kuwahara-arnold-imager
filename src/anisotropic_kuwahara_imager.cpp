#include <vector>

#include <ai.h>
#include <omp.h>

#include "kuwahara_arnold.hpp"
#include "structure_tensor.hpp"

#pragma warning(disable: 4100)


AI_IMAGER_NODE_EXPORT_METHODS(AnisotropicKuwaharaImagerMtd);

// polynomialBase(u, v) = max(0, u^2 - eta*v^2)^2
inline float polynomialBase(float u, float v, float eta = 0.4f)
{
    float val = (u * u) - (eta * v * v);
    return (val > 0.0f) ? (val * val) : 0.0f;
}

inline float polynomialSector(float u, float v, int sectorIndex, int totalSectors, float eta = 0.4f)
{
    // angle = -2πi / N
    float angle = -AI_PITIMES2 * static_cast<float>(sectorIndex) / static_cast<float>(totalSectors);
    float cosA = std::cos(angle);
    float sinA = std::sin(angle);

    // Rotation inverse
    float uR = cosA * u - sinA * v;
    float vR = sinA * u + cosA * v;

    // Évaluer la fonction polynomiale de base
    return polynomialBase(uR, vR, eta);
}




// float sectorWeight(
//     int i,            // index du secteur (0..N-1)
//     float u, float v, // coordonnées dans le disque unité
//     int N,            // nombre de secteurs
//     float sigmaAngle, // écart-type pour le lissage angulaire
//     float sigmaRad    // écart-type pour la décroissance radiale
// )
// {
//     // Coordonnées polaires
//     float r = std::sqrt(u*u + v*v);
//     float angle = std::atan2(v, u);
//     if (angle < 0) angle += (2.0f * AI_PI);

//     // Centre du secteur i
//     float sectorCenter = (i + 0.5f) * (2.0f * AI_PI / N);

//     // Différence angulaire (on s'assure qu'elle est dans [0, π])
//     float dphi = std::fabs(angle - sectorCenter);
//     if (dphi > AI_PI)
//         dphi = 2.0f * AI_PI - dphi;

//     // Gaussienne angulaire
//     float angularWeight = std::exp(- (dphi * dphi) / (2.0f * sigmaAngle * sigmaAngle));

//     // Gaussienne radiale
//     float radialWeight = std::exp(- (r * r) / (2.0f * sigmaRad * sigmaRad));

//     return angularWeight * radialWeight;
// }



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
    const void*             bucket_data;
    int                     type;
    std::vector<PixelData>  pixels_data;

    AOVData(const void* bucket_data, int type) 
        : bucket_data(bucket_data), type(type) {}
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
    // Set the imager schedule type to full frame always as 
    // we need access to all neighbouring pixels
    schedule = AtImagerSchedule::FULL_FRAME;
}

imager_evaluate
{
    // Node Parameters
    const int radius = AiNodeGetInt(node, radius_str);
    if (radius < 1)
        return;

    const float radiusf = static_cast<float>(radius);

    const float ellipse_min_radius = AiMax(1.0f, AI_EPSILON);
    const int sectors_num = 8;  // N
    const float sectors_numf = static_cast<float>(sectors_num);
    
    grid::GridSize grid(bucket_size_x, bucket_size_y);

    const int num_pixels = bucket_size_x * bucket_size_y;
    std::vector<AtRGBA> data(num_pixels, {AI_RGB_BLACK, 1.0f});

    // Iterator vars
    int aov_type = 0;
    const void *bucket_data;
    AtString output_name;

    // const int y = 400;
    // const int x = 400;
    // const int base_idx = y * bucket_size_x;

    
    while (AiOutputIteratorGetNext(iterator, &output_name, &aov_type, &bucket_data))
    {
        auto aov_data = AOVData(bucket_data, aov_type);
        AtRGBA* rgba = (AtRGBA*)bucket_data;

        const cv::Mat structureTensor = structure_tensor::computeStructureTensor(rgba, grid, 3, 1.0);

        // Iterate over the bucket
        #pragma omp parallel for
        for (int y = 0; y < bucket_size_y; ++y)
        {
            const int base_idx = y * bucket_size_x;

            for (int x = 0; x < bucket_size_x; ++x)
            {
                // Index of the pixel in the bucket
                const int idx = base_idx + x;

                const cv::Point2f center_point(static_cast<float>(x), static_cast<float>(y));

                // Compute local orientation and anisotropy
                // The local orientation is in radians
                const auto [orientation_rad, anisotropy] = 
                    structure_tensor::computeLocalOrientationAndAnisotropyAtPoint(structureTensor, x, y);

                // Compute the kernel shape
                float ellipse_major_radius = ((ellipse_min_radius + anisotropy) / ellipse_min_radius) * radiusf;  // a
                ellipse_major_radius = AiMax(radiusf, AiMin(ellipse_major_radius, 2.f * radiusf));  // r <= a <= 2r
            
                float ellipse_minor_radius = (ellipse_min_radius / (ellipse_min_radius + anisotropy)) * radiusf;  // b
                ellipse_minor_radius = AiMax(radiusf * .5f, AiMin(ellipse_minor_radius, radiusf));

                // Compute the scale matrix
                cv::Matx22f scale_mat(1.0f / ellipse_major_radius, .0f, .0f, 1.0f / ellipse_minor_radius);  // S = [1/a 0; 0 1/b]

                // Compute the rotation matrix
                const float orientation_cos = std::cos(orientation_rad);
                const float orientation_sin = std::sin(orientation_rad);
                cv::Matx22f rot_mat(orientation_cos, orientation_sin, -orientation_sin, orientation_cos);  // R = [cos(theta) sin(theta); -sin(theta) cos(theta)]

                // Compute the inverse affine matrix
                cv::Matx22f affine_mat = scale_mat * rot_mat ;  // M = S * R
                cv::Matx22f inv_affine_mat = affine_mat.inv();  // M^-1
            
                // Compute the bounding box of the ellipse
                const float half_width = 
                    std::ceil(std::abs(ellipse_major_radius * orientation_cos) + 
                    std::abs(ellipse_minor_radius * orientation_sin));

                const float half_height =
                    std::ceil(std::abs(ellipse_major_radius * orientation_sin) +
                    std::abs(ellipse_minor_radius * orientation_cos));

                // Définir le patch en fonction de la boîte englobante
                const int patch_min_x = AiMax(x - static_cast<int>(half_width), 0);
                const int patch_max_x = AiMin(x + static_cast<int>(half_width), bucket_size_x);
                const int patch_min_y = AiMax(y - static_cast<int>(half_height), 0);
                const int patch_max_y = AiMin(y + static_cast<int>(half_height), bucket_size_y);


                std::array<float, 8>    sumWeight       = {}; sumWeight.fill(0.0f);
                std::array<AtRGBA, 8>   sumColor        = {}; sumColor.fill(AI_RGB_BLACK);
                std::array<float, 8>    sumIntensity    = {}; sumIntensity.fill(0.0f);
                std::array<float, 8>    sumIntensitySq  = {}; sumIntensitySq.fill(0.0f);


                // Iterate over the pixels in the patch
                for (int py = patch_min_y; py < patch_max_y; ++py)
                {
                    const int base_patch_idx = py * bucket_size_x;

                    for (int px = patch_min_x; px < patch_max_x; ++px)
                    {
                        const int patch_idx = base_patch_idx + px;

                        // Compute relative coordinates
                        const cv::Point2f patch_point(static_cast<float>(px), static_cast<float>(py));
                        const cv::Point2f local_point = patch_point - center_point;

                        // (u, v) = M * local_point
                        const cv::Point2f disc_point = affine_mat * local_point;
                        const float &u = disc_point.x;
                        const float &v = disc_point.y;
                        
                        const bool is_inside = (u*u + v*v) <= 1.0f;
                        if (!is_inside)
                            continue;

                        // Calculer la couleur (R, G, B) et intensité
                        const AtRGBA& pixel_color = rgba[patch_idx];
                        const float I = (0.2126f * pixel_color.r) + (0.7152f * pixel_color.g) + (0.0722f * pixel_color.b);

                        // Calcul des poids polynomiaux pour chaque secteur
                        std::vector<float> sectorWeights(sectors_num, 0.0f);
                        float sumW = 0.0f;
                        float eta = 0.4f; // paramètre ajustable

                        for (int i = 0; i < sectors_num; ++i)
                        {
                            float w_i = polynomialSector(u, v, i, sectors_num, eta);
                            sectorWeights[i] = w_i;
                            sumW += w_i;
                        }

                        if (sumW <= AI_EPSILON)
                            continue;
          
                        for (int i = 0; i < sectors_num; ++i)
                        {
                            sectorWeights[i] /= sumW;

                            // Accumuler pour chaque secteur
                            sumWeight[i]     += sectorWeights[i];
                            sumColor[i]      += pixel_color * sectorWeights[i];
                            sumIntensity[i]  += I * sectorWeights[i];
                            sumIntensitySq[i]+= (I*I) * sectorWeights[i];
                        }
                        
                    }
                }

                // 2B) Après le patch, on calcule la couleur finale
                AtRGBA color_total = AI_RGB_BLACK;
                float alpha_total = 0.0f;

                float k = 10.0f;
                int q = 8;

                for (int i = 0; i < sectors_num; ++i)
                {
                    if (sumWeight[i] <= AI_EPSILON) 
                        continue;

                    AtRGBA meanColor_i = sumColor[i] / sumWeight[i];
                    float meanI = sumIntensity[i] / sumWeight[i];
                    float meanI2= sumIntensitySq[i] / sumWeight[i];
                    float var_i = meanI2 - meanI*meanI;

                    float alpha_i = 1.0f / (1.0f + k * std::pow(var_i, (float)q));
                    color_total += meanColor_i * alpha_i;
                    alpha_total += alpha_i;
                }

                // cv::Vec3f finalColor(0,0,0);
                // if (alpha_total > AI_EPSILON)
                //     finalColor = color_total / alpha_total;

                


                // data[idx] = color_total / 8.0f;
                data[idx] = color_total / alpha_total;
                // data[idx].a = 1.0f;

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
