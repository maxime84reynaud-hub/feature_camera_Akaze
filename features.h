#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>

#if CV_VERSION_MAJOR >= 5
    #include <opencv2/features.hpp>
    #include <opencv2/geometry.hpp>
    #include <opencv2/xfeatures2d.hpp>
    using AkazeDetector = cv::xfeatures2d::AKAZE;
#else
    #include <opencv2/features2d.hpp>
    #include <opencv2/calib3d.hpp>
    using AkazeDetector = cv::AKAZE;
#endif

#include <string>
#include <vector>

struct ResultatDetection {
    bool carteTrouvee = false;
    double x = 0.0;
    double y = 0.0;
    double thetaDeg = 0.0;
    int nbFeaturesReconnues = 0;
    std::vector<cv::KeyPoint> inliersCamera;
    std::vector<cv::Point2f> coinsProjetes;
};

class DetecteurAKAZE {
public:
    explicit DetecteurAKAZE(float seuilAKAZE = 0.001f);

    cv::Mat pretraiter(const cv::Mat& imageBGR) const;
    bool genererCarteReference(const cv::Mat& imageRefBGR, const std::string& fichierYml, cv::Mat& imageVisuOut);
    bool chargerCarteReference(const std::string& fichierYml);
    ResultatDetection analyserFrame(const cv::Mat& frameBGR, cv::Mat& affichageOut) const;

    int getNbFeaturesCarte() const { return static_cast<int>(refKeypoints_.size()); }

private:
    cv::Ptr<cv::CLAHE> clahe_;
    cv::Ptr<AkazeDetector> akaze_;
    cv::BFMatcher matcher_;

    int refWidth_ = 0;
    int refHeight_ = 0;
    std::vector<cv::KeyPoint> refKeypoints_;
    cv::Mat refDescriptors_;
};

#endif // FEATURES_H