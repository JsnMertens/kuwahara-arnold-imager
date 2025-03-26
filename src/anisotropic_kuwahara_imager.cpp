#include <vector>

#include <ai.h>

#include "kuwahara_arnold.hpp"
#include "structure_tensor.hpp"


AI_IMAGER_NODE_EXPORT_METHODS(AnisotropicKuwaharaImagerMtd);



float sectorWeight(
    int i,            // index du secteur (0..N-1)
    float u, float v, // coordonnées dans le disque unité
    int N,            // nombre de secteurs
    float sigmaAngle, // écart-type pour le lissage angulaire
    float sigmaRad    // écart-type pour la décroissance radiale
)
{
    // Coordonnées polaires
    float r = std::sqrt(u*u + v*v);
    float angle = std::atan2(v, u);
    if (angle < 0) angle += 2.0f * CV_PI;

    // Centre du secteur i
    float sectorCenter = (i + 0.5f) * (2.0f * CV_PI / N);

    // Différence angulaire (on s'assure qu'elle est dans [0, π])
    float dphi = std::fabs(angle - sectorCenter);
    if (dphi > CV_PI)
        dphi = 2.0f * CV_PI - dphi;

    // Gaussienne angulaire
    float angularWeight = std::exp(- (dphi * dphi) / (2.0f * sigmaAngle * sigmaAngle));

    // Gaussienne radiale
    float radialWeight = std::exp(- (r * r) / (2.0f * sigmaRad * sigmaRad));

    return angularWeight * radialWeight;
}



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
    const float radiusf = static_cast<float>(radius);

    const float elongation_factor = 1.0f;
    const float ellipse_min_radius = AiMax(1.0f, AI_EPSILON);
    const size_t sectors_num = 8;  // N
    
    std::vector<AOVData> aovs;
    grid::GridSize grid(bucket_size_x, bucket_size_y);

    const int num_pixels = bucket_size_x * bucket_size_y;
    std::vector<AtRGBA> data(num_pixels, {AI_RGB_BLACK, .1f});

    // Iterator vars
    int aov_type = 0;
    const void *bucket_data;
    AtString output_name;


    float sigmaAngle = 0.4f;  // ex. angle smoothing
    float sigmaRad   = 0.4f;  // ex. radial decay
    int N = 8;               // nombre de secteurs

    // Paramètres pour la combinaison finale
    float k = 1.0f;   // influence de la variance
    int q = 8;        // exponent pour (s_i)^q

    std::vector<float> sumWeight(N, 0.0f);       // somme des poids w_i
    std::vector<cv::Vec3f> sumColor(N, cv::Vec3f(0,0,0));  // somme de (R, G, B)*w_i
    std::vector<float> sumIntensity(N, 0.0f);    // somme de intensité*w_i
    std::vector<float> sumIntensitySq(N, 0.0f);  // somme de (intensité^2)*w_i


    // const int y = 249;
    // const int x = 249;
    // const int base_idx = y * bucket_size_x;

    while (AiOutputIteratorGetNext(iterator, &output_name, &aov_type, &bucket_data))
    {
        auto aov_data = AOVData(bucket_data, aov_type);
        AtRGBA* rgba = (AtRGBA*)bucket_data;

        const cv::Mat structureTensor = structure_tensor::ComputeStructureTensor(rgba, grid, 3, 1.0);

        // Iterate over the bucket
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
                auto [orientation_rad, anisotropy] = 
                    structure_tensor::ComputeLocalOrientationAndAnisotropyAtPoint(structureTensor, x, y);

                // Compute the scale matrix
                float ellipse_major_radius = ((ellipse_min_radius + anisotropy) / ellipse_min_radius) * radius;  // a
                ellipse_major_radius = AiMax(radiusf, AiMin(ellipse_major_radius, 2.f * radiusf));  // r <= a <= 2r

                float ellipse_minor_radius = (ellipse_min_radius / (ellipse_min_radius + anisotropy)) * radius;  // b
                ellipse_minor_radius = AiMax(radiusf * .5f, AiMin(ellipse_minor_radius, radiusf));  // r/2 <= b <= r

                cv::Matx22f scale_mat(1.0f / ellipse_major_radius, .0f, .0f, 1.0f / ellipse_minor_radius);  // S = [1/a 0; 0 1/b]

                // Compute the rotation matrix
                const float cos_theta = std::cos(orientation_rad);
                const float sin_theta = std::sin(orientation_rad);
                cv::Matx22f rot_mat(cos_theta, sin_theta, -sin_theta, cos_theta);  // R = [cos(theta) sin(theta); -sin(theta) cos(theta)]

                // Compute the inverse affine matrix
                cv::Matx22f affine_mat = scale_mat * rot_mat ;  // M = S * R
                cv::Matx22f inv_affine_mat = affine_mat.inv();  // M^-1
                
                const int patch_min_x = AiMax(x - radius, 0);
                const int patch_max_x = AiMin(x + radius, bucket_size_x);
                const int patch_min_y = AiMax(y - radius, 0);
                const int patch_max_y = AiMin(y + radius, bucket_size_y);

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

                        const bool is_inside = (u * u + v * v) <= 1.0f;
                        if (!is_inside)
                            continue;

                        for (int i = 0; i < sectors_num; ++i)
                        {
                            const float weight = sectorWeight(i, u, v, sectors_num, sigmaAngle, sigmaRad);
                            sumWeight[i] += weight;
                            sumColor[i] += cv::Vec3f(rgba[patch_idx].r, rgba[patch_idx].g, rgba[patch_idx].b) * weight;
                            sumIntensity[i] += (0.299f * rgba[patch_idx].r + 0.587f * rgba[patch_idx].g + 0.114f * rgba[patch_idx].b) * weight;
                            sumIntensitySq[i] += (0.299f * rgba[patch_idx].r + 0.587f * rgba[patch_idx].g + 0.114f * rgba[patch_idx].b) * (0.299f * rgba[patch_idx].r + 0.587f * rgba[patch_idx].g + 0.114f * rgba[patch_idx].b) * weight;
                        } 


                        // std::cout << "center_point x: " << center_point.x << " y: " << center_point.y << std::endl;
                        // std::cout << "patch_point x: " << patch_point.x << " y: " << patch_point.y << std::endl;
                        // std::cout << "local_point x: " << local_point.x << " y: " << local_point.y << std::endl;
                        // std::cout << "disc_point x: " << disc_point.x << " y: " << disc_point.y << std::endl;
                        // std::cout << "---" << std::endl;


                        // // Récupérer la couleur du pixel (px, py)
                        // float r = rgba[patch_idx].r;
                        // float g = rgba[patch_idx].g;
                        // float b = rgba[patch_idx].b;
                        // // Calcul d'une intensité (ex: luminance)
                        // float intensity = 0.299f*r + 0.587f*g + 0.114f*b;

                        



                        // // Calcul des poids pour chaque secteur
                        // std::vector<float> weights(N, 0.0f);
                        // float sumW = 0.0f;
                        // for (int i = 0; i < N; ++i)
                        // {
                        //     float w_i = sectorWeight(i, u, v, N, sigmaAngle, sigmaRad);
                        //     weights[i] = w_i;
                        //     sumW += w_i;
                        // }
                        // // Normaliser
                        // if (sumW > 1e-6f) {
                        //     for (int i = 0; i < N; ++i) {
                        //         weights[i] /= sumW;
                        //     }
                        // }

                        // // Accumuler pour chaque secteur
                        // for (int i = 0; i < N; ++i)
                        // {
                        //     float w_i = weights[i];
                        //     sumWeight[i] += w_i;
                        //     sumColor[i] += cv::Vec3f(r, g, b) * w_i;
                        //     sumIntensity[i] += intensity * w_i;
                        //     sumIntensitySq[i] += (intensity * intensity) * w_i;
                        // }
                    }
                }

                // // 6) Calculer moyennes et variances dans chaque secteur
                // std::vector<cv::Vec3f> meanColor(N);
                // std::vector<float> varI(N, 0.0f); // variance intensité
                // for (int i = 0; i < N; ++i)
                // {
                //     if (sumWeight[i] > 1e-6f) {
                //         meanColor[i] = sumColor[i] / sumWeight[i];
                //         float meanI = sumIntensity[i] / sumWeight[i];
                //         float secondMoment = sumIntensitySq[i] / sumWeight[i];
                //         varI[i] = secondMoment - meanI * meanI; // variance
                //     } else {
                //         meanColor[i] = cv::Vec3f(0,0,0);
                //         varI[i] = 0.0f;
                //     }
                // }

                // // 7) Combiner les secteurs
                // //    alpha_i = 1 / (1 + k * (s_i)^q)
                // //    où s_i = sqrt(varI[i])
                // std::vector<float> alpha(N, 0.0f);
                // float sumAlpha = 0.0f;
                // for (int i = 0; i < N; ++i)
                // {
                //     float s_i = std::sqrt(varI[i]);
                //     alpha[i] = 1.0f / (1.0f + k * std::pow(s_i, q));
                //     sumAlpha += alpha[i];
                // }

                // cv::Vec3f finalColor(0,0,0);
                // if (sumAlpha > 1e-6f) {
                //     for (int i = 0; i < N; ++i) {
                //         finalColor += alpha[i] * meanColor[i];
                //     }
                //     finalColor /= sumAlpha;
                // }

                // // 8) Assigner la couleur finale au pixel (x, y)
                // rgba[idx].r = finalColor[0];
                // rgba[idx].g = finalColor[1];
                // rgba[idx].b = finalColor[2];
                // rgba[idx].a = 1.0f; // alpha = 1.0f par exemple
            
            

                
                // for (int i = 0; i < sectors_num; ++i) {
                //     float sector_center_rad = (i + 0.5f) * (2 * AI_PI / sectors_num);  // in radians
                // }
                
                
                // float elongation = 1.0f + eccentricity * elongation_factor;

                // grid::GridPoint center(x, y);

                // AtRGBA  best_color = rgba[idx];
                // float   best_variance = AI_BIG;

                // AtRGBA  mean_color = AI_RGBA_ZERO;
                // float   variance = 0.0f;

                // // Iterate over the 4 quadrants
                // for (auto quadrant : kuwahara_arnold::quadrants)
                // {
                //     grid::GridRegion quadrant_region = kuwahara_arnold::ComputeQuadrantRegion(center, grid, radius, quadrant);
                //     kuwahara_arnold::ComputeRegion(rgba, grid, quadrant_region, mean_color, variance);

                //     if (variance < best_variance)
                //     {
                //         best_variance = variance;
                //         best_color = mean_color;
                //     }
                // }

                // // Store pixel data for each index
                // aov_data.pixels_data.push_back(PixelData(idx, best_color));





            }
        }


    //     // aovs.push_back(aov_data);
    // }

    // Set the output color for each AOV
    for (int y = 0; y < bucket_size_y; ++y)
    {
        const int base_idx = y * bucket_size_x;

        for (int x = 0; x < bucket_size_x; ++x)
        {
            const int idx = base_idx + x;
            rgba[idx] = data[idx];
        }
    }
}}

node_finish
{
}
