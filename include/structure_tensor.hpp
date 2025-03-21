#ifndef KUWAHARA_ARNOLD_STRUCTURE_TENSOR_H_
#define KUWAHARA_ARNOLD_STRUCTURE_TENSOR_H_

#include <opencv2/opencv.hpp>


namespace
{

template<typename color_type>
inline float RGBtoLuminance(const color_type& pixel)
{
    return 0.299f * pixel.r + 0.587f * pixel.g + 0.114f * pixel.b;
}

void BuildImgLuminance(const AtRGBA* rgba, int bucket_size_x, int bucket_size_y, cv::Mat& img)
{
    for (int y = 0; y < bucket_size_y; ++y)
    {
        for (int x = 0; x < bucket_size_x; ++x)
        {
            int idx = y * bucket_size_x + x;
            AtRGBA pixel = rgba[idx];

            // Convert to luminance
            float luminance = RGBtoLuminance(pixel);

            img.at<float>(y, x) = luminance;
        }
    }
}

} // anonymous namespace

namespace structure_tensor
{

inline cv::Mat computeStructureTensor(const AtRGBA* rgba, int bucket_size_x, int bucket_size_y, int ksize = 3, double sigma = 1.0)
{
    // Convertir l'image en niveaux de gris
    cv::Mat img(bucket_size_y, bucket_size_x, CV_32F);
    BuildImgLuminance(rgba, bucket_size_x, bucket_size_y, img);

    // Calcul des gradients sur x et y
    cv::Mat gradX, gradY;
    cv::Sobel(img, gradX, CV_32F, 1, 0, ksize);
    cv::Sobel(img, gradY, CV_32F, 0, 1, ksize);

    // Calcul des produits de gradients
    cv::Mat gradXX = gradX.mul(gradX);
    cv::Mat gradYY = gradY.mul(gradY);
    cv::Mat gradXY = gradX.mul(gradY);

    // Lissage des composantes pour une meilleure robustesse
    cv::GaussianBlur(gradXX, gradXX, cv::Size(ksize, ksize), sigma);
    cv::GaussianBlur(gradYY, gradYY, cv::Size(ksize, ksize), sigma);
    cv::GaussianBlur(gradXY, gradXY, cv::Size(ksize, ksize), sigma);

    // Fusionner les trois composantes en une image à 3 canaux
    std::vector<cv::Mat> channels = { gradXX, gradXY, gradYY };
    cv::Mat structureTensor;
    cv::merge(channels, structureTensor);

    return structureTensor;
}

} // namespace structure_tensor

#endif // KUWAHARA_ARNOLD_STRUCTURE_TENSOR_H_
