#ifndef KUWAHARA_ARNOLD_STRUCTURE_TENSOR_H_
#define KUWAHARA_ARNOLD_STRUCTURE_TENSOR_H_

#include <omp.h>

#include <opencv2/opencv.hpp>

namespace
{

void buildLuminanceImage(const AtRGBA* rgba, const int bucket_size_x, const int bucket_size_y, cv::Mat& img)
{
    #pragma omp parallel for
    for (int y = 0; y < bucket_size_y; ++y)
    {
        float* img_row_ptr = img.ptr<float>(y);
        int base_idx = y * bucket_size_x;

        for (int x = 0; x < bucket_size_x; ++x)
        {
            const int idx = base_idx + x;
            const AtRGBA &pixel = rgba[idx];

            // Convert to luminance
            const float luminance = 0.299f * pixel.r + 0.587f * pixel.g + 0.114f * pixel.b;
            img_row_ptr[x] = luminance;
        }
    }
}

} // anonymous namespace

namespace structure_tensor
{

inline
cv::Mat computeStructureTensor(
    const AtRGBA* color_ptr,
    const cv::Point2i& img_size,
    unsigned int kernelsize = 3
)
{
    // Build luminance image
    cv::Mat img(img_size.y, img_size.x, CV_32F);
    buildLuminanceImage(color_ptr, img_size.x, img_size.y, img);

    // Compute gradients
    cv::Mat grad_x, grad_y;
    cv::Sobel(img, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(img, grad_y, CV_32F, 0, 1, 3);

    // Compute structure tensor
    // Structure Tensor = J = [Jxx Jxy; Jxy Jyy]
    cv::Mat Jxx = grad_x.mul(grad_x);
    cv::Mat Jxy = grad_x.mul(grad_y);
    cv::Mat Jyy = grad_y.mul(grad_y);

    // Merge components into a 3-channel image
    std::vector<cv::Mat> channels = { Jxx, Jxy, Jyy };
    cv::Mat structure_tensor;
    cv::merge(channels, structure_tensor);

    // Blur the structure tensor for noise reduction
    // The kernel size should be odd and positive, force it to be odd
    if (kernelsize % 2 == 0)
        kernelsize += 1;
        
    cv::Size size(kernelsize, kernelsize);
    cv::GaussianBlur(structure_tensor, structure_tensor, size, 0.0);

    return structure_tensor;
}

inline
std::pair<float, float> computeLocalOrientationAndAnisotropyAtPoint(const cv::Mat &structuretensor, const int x, const int y)
{
    // Retrieve the structure tensor values at the given pixel
    const cv::Vec3f &T = structuretensor.at<cv::Vec3f>(y, x);
    const float &Jxx = T[0];
    const float &Jxy = T[1];
    const float &Jyy = T[2];

    // Build the 2x2 matrix
    const cv::Matx22f T_mat(Jxx, Jxy, Jxy, Jyy);

    // Compute eigenvalues and eigenvectors
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(T_mat, eigenvalues, eigenvectors);

    const float &lambda1 = eigenvalues.at<float>(0); // Biggest eigenvalue
    const float &lambda2 = eigenvalues.at<float>(1); // Smallest eigenvalue

    // Compute anisotropy : A = (lambda1 - lambda2) / (lambda1 + lambda2)
    float anisotropy = ((lambda1 + lambda2) > AI_EPSILON) ? (lambda1 - lambda2) / (lambda1 + lambda2) : 0;

    // Compute local orientation, use the eigenvector associated with the smallest value (row 1)
    const cv::Vec2f &minor_eigenvector = eigenvectors.at<cv::Vec2f>(1);
    float orientation = std::atan2(minor_eigenvector[1], minor_eigenvector[0]);  // in radians

    return {orientation, anisotropy};
}

} // namespace structure_tensor

#endif // KUWAHARA_ARNOLD_STRUCTURE_TENSOR_H_
