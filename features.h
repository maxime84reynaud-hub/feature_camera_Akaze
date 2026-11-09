#ifndef FEATURES_H
#define FEATURES_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

// Compatibilité automatique entre OpenCV 5 (Windows MSYS2) et OpenCV 4 (Linux)
#if CV_VERSION_MAJOR >= 5
#include <opencv2/xfeatures2d.hpp>
using AKAZEAlgo = cv::xfeatures2d::AKAZE;
#else
using AKAZEAlgo = cv::AKAZE;
#endif

#include <string>
#include <vector>

struct ResultatDetection {
    bool carteTrouvee = false;
    double x = 0.0;
    double y = 0.0;
    double thetaDeg = 0.0;
    int nbMatchesBruts = 0;
    int nbFeaturesReconnues = 0;
    int nbPointsMouvement = 0;               // Nombre de points rouges détectés entre 2 frames
    std::vector<cv::KeyPoint> inliersCamera;
    std::vector<cv::Point2f> coinsProjetes;
    std::vector<cv::Point> pointsMouvement;  // Coordonnées des différences entre t-1 et t
};

class DetecteurAKAZE {
public:
    explicit DetecteurAKAZE(float seuilAKAZE = 0.001f);

    cv::Mat pretraiter(const cv::Mat& imageBGR) const;
    bool genererCarteReference(const cv::Mat& imageRefBGR, const std::string& fichierYml, cv::Mat& imageVisuOut);
    bool chargerCarteReference(const std::string& fichierYml);

    // Analyse complète : Mouvement (points rouges) + Reconnaissance AKAZE + Pose (X, Y, Theta)
    ResultatDetection analyserFrame(const cv::Mat& frameBGR, cv::Mat& affichageOut);

    // Compare la frame actuelle avec la frame précédente et trace les points rouges
    void detecterMouvement(const cv::Mat& frameBGR, cv::Mat& affichageOut, ResultatDetection& res, double seuilDiff = 25.0);

    int getNbFeaturesCarte() const { return static_cast<int>(refKeypoints_.size()); }

private:
    cv::Ptr<cv::CLAHE> clahe_;
    cv::Ptr<AKAZEAlgo> akaze_;
    cv::BFMatcher matcher_;

    int refWidth_ = 0;
    int refHeight_ = 0;
    std::vector<cv::KeyPoint> refKeypoints_;
    cv::Mat refDescriptors_;

    cv::Mat prevFrameGris_; // Stocke l'image t-1 pour la comparaison de mouvement
};

#endif // FEATURES_H