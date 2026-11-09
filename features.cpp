#include "features.h"
#include <cmath>
#include <iostream>

DetecteurAKAZE::DetecteurAKAZE(float seuilAKAZE)
    : matcher_(cv::NORM_HAMMING) {
    clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));
    akaze_ = AKAZEAlgo::create(
        AKAZEAlgo::DESCRIPTOR_MLDB,
        0, 3,
        seuilAKAZE,
        4, 4
    );
}

cv::Mat DetecteurAKAZE::pretraiter(const cv::Mat& imageBGR) const {
    cv::Mat gris, resultat;
    if (imageBGR.channels() == 3) {
        cv::cvtColor(imageBGR, gris, cv::COLOR_BGR2GRAY);
    } else {
        gris = imageBGR;
    }
    clahe_->apply(gris, resultat); // égalisation de la lumière pour faire ressortir des détails 
    return resultat;
}

bool DetecteurAKAZE::genererCarteReference(const cv::Mat& imageRefBGR, const std::string& fichierYml, cv::Mat& imageVisuOut) {
    if (imageRefBGR.empty()) return false;

    cv::Mat pretraitee = pretraiter(imageRefBGR);
    refKeypoints_.clear();
    refDescriptors_.release();

    akaze_->detectAndCompute(pretraitee, cv::noArray(), refKeypoints_, refDescriptors_);
    if (refKeypoints_.size() < 8) {
        std::cerr << "Erreur : pas assez de features (" << refKeypoints_.size() << ")." << std::endl;
        return false;
    }

    refWidth_ = pretraitee.cols;
    refHeight_ = pretraitee.rows;

    // Sauvegarde dans le fichier YML
    cv::FileStorage fs(fichierYml, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "width" << refWidth_;
    fs << "height" << refHeight_;
    fs << "keypoints" << refKeypoints_;
    fs << "descriptors" << refDescriptors_;
    fs.release();

    // Images de visu où sont les features 
    cv::cvtColor(pretraitee, imageVisuOut, cv::COLOR_GRAY2BGR);
    cv::drawKeypoints(
        imageVisuOut,
        refKeypoints_,
        imageVisuOut,
        cv::Scalar(0, 255, 0),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );

    return true;
}

bool DetecteurAKAZE::chargerCarteReference(const std::string& fichierYml) {
    cv::FileStorage fs(fichierYml, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;

    fs["width"] >> refWidth_;
    fs["height"] >> refHeight_;
    fs["keypoints"] >> refKeypoints_;
    fs["descriptors"] >> refDescriptors_;
    fs.release();

    return !refDescriptors_.empty() && refKeypoints_.size() >= 8;
}

ResultatDetection DetecteurAKAZE::analyserFrame(const cv::Mat& frameBGR, cv::Mat& affichageOut) {
    ResultatDetection res;
    affichageOut = frameBGR.clone();

    if (frameBGR.empty()) return res;

    // comparaison de mvt entre deux images 
    detecterMouvement(frameBGR, affichageOut, res, 25.0);

    if (refDescriptors_.empty()) return res;

    // détection des features 
    cv::Mat framePretraitee = pretraiter(frameBGR);
    std::vector<cv::KeyPoint> frameKeypoints;
    cv::Mat frameDescriptors;
    akaze_->detectAndCompute(framePretraitee, cv::noArray(), frameKeypoints, frameDescriptors);

    if (frameDescriptors.empty() || frameKeypoints.size() < 4) return res;

    // Appariement k-NN (k = 2) et test du ratio de Lowe
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher_.knnMatch(refDescriptors_, frameDescriptors, knnMatches, 2);

    const float ratioLowe = 0.75f;
    std::vector<cv::DMatch> bonsMatches;
    std::vector<cv::Point2f> ptsRef, ptsCamera;

    for (const auto& matchPair : knnMatches) {
        if (matchPair.size() == 2 && matchPair[0].distance < ratioLowe * matchPair[1].distance) {
            bonsMatches.push_back(matchPair[0]);
            ptsRef.push_back(refKeypoints_[matchPair[0].queryIdx].pt);
            ptsCamera.push_back(frameKeypoints[matchPair[0].trainIdx].pt);
        }
    }

    res.nbMatchesBruts = static_cast<int>(bonsMatches.size());

    // Filtrage géométrique RANSAC et calcul de l'homographie
    if (bonsMatches.size() >= 8) {
        std::vector<uchar> inliersMask;
        cv::Mat H = cv::findHomography(ptsRef, ptsCamera, cv::RANSAC, 3.0, inliersMask);

        for (size_t i = 0; i < inliersMask.size(); i++) {
            if (inliersMask[i]) {
                res.inliersCamera.push_back(frameKeypoints[bonsMatches[i].trainIdx]);
            }
        }
        res.nbFeaturesReconnues = static_cast<int>(res.inliersCamera.size());

        // features détecter frame vert
        cv::drawKeypoints(
            affichageOut,
            res.inliersCamera,
            affichageOut,
            cv::Scalar(0, 255, 0),
            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
        );

        // si on trouve des features alors on calcul positon 
        if (!H.empty() && res.nbFeaturesReconnues >= 8) {
            res.carteTrouvee = true;

            float cx = refWidth_ * 0.5f;
            float cy = refHeight_ * 0.5f;
            std::vector<cv::Point2f> ptsRepereRef = {
                cv::Point2f(0.0f, 0.0f),
                cv::Point2f(static_cast<float>(refWidth_), 0.0f),
                cv::Point2f(static_cast<float>(refWidth_), static_cast<float>(refHeight_)),
                cv::Point2f(0.0f, static_cast<float>(refHeight_)),
                cv::Point2f(cx, cy),          // Index 4 : Centre de la carte
                cv::Point2f(cx + 100.0f, cy)  // Index 5 : Vecteur axe X pour l'angle Theta
            };

            std::vector<cv::Point2f> ptsProjetes;
            cv::perspectiveTransform(ptsRepereRef, ptsProjetes, H);

            res.coinsProjetes.assign(ptsProjetes.begin(), ptsProjetes.begin() + 4);
            res.x = ptsProjetes[4].x;
            res.y = ptsProjetes[4].y;

            double dx = ptsProjetes[5].x - ptsProjetes[4].x;
            double dy = ptsProjetes[5].y - ptsProjetes[4].y;
            res.thetaDeg = std::atan2(dy, dx) * 180.0 / CV_PI;

            // cadre + vecteur orientation 
            for (int i = 0; i < 4; i++) {
                cv::line(affichageOut, res.coinsProjetes[i], res.coinsProjetes[(i + 1) % 4], cv::Scalar(255, 0, 0), 2);
            }
            cv::circle(affichageOut, ptsProjetes[4], 5, cv::Scalar(0, 0, 255), -1);
            cv::arrowedLine(affichageOut, ptsProjetes[4], ptsProjetes[5], cv::Scalar(0, 0, 255), 2);
        }
    }

    return res;
}

void DetecteurAKAZE::detecterMouvement(const cv::Mat& frameBGR, cv::Mat& affichageOut, ResultatDetection& res, double seuilDiff) {
    cv::Mat grisActuel;
    if (frameBGR.channels() == 3) {
        cv::cvtColor(frameBGR, grisActuel, cv::COLOR_BGR2GRAY);
    } else {
        grisActuel = frameBGR.clone();
    }

    // problème de bruit application d'un flou gaussien y'avait plein de faux positifs 
    cv::GaussianBlur(grisActuel, grisActuel, cv::Size(5, 5), 0);

    if (!prevFrameGris_.empty() && prevFrameGris_.size() == grisActuel.size()) {
        cv::Mat diff, masqueMouvement;

        // on fait la différenre entre les 2 images
        cv::absdiff(prevFrameGris_, grisActuel, diff);

        // on regarde ce qui change 
        cv::threshold(diff, masqueMouvement, seuilDiff, 255, cv::THRESH_BINARY);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(masqueMouvement, masqueMouvement, cv::MORPH_OPEN, kernel);

        // on créer notre grille de pont rouge zones mobile 
        const int pasPoint = 8; // 1 pt rouge tt les 8 pixels
        for (int y = 0; y < masqueMouvement.rows; y += pasPoint) {
            const uchar* ligne = masqueMouvement.ptr<uchar>(y);
            for (int x = 0; x < masqueMouvement.cols; x += pasPoint) {
                if (ligne[x] > 0) {
                    cv::Point pt(x, y);
                    res.pointsMouvement.push_back(pt);
                    // Tracé d'un point rouge vif (BGR : 0, 0, 255)
                    cv::circle(affichageOut, pt, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
                }
            }
        }
        res.nbPointsMouvement = static_cast<int>(res.pointsMouvement.size());
    }

    // on save l'image traiter pour la comparer a la suivante 
    prevFrameGris_ = grisActuel.clone();
}