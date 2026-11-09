#include "features.h"
#include <iostream>
#include <filesystem>

constexpr double TABLE_X_MM = 3000.0;
constexpr double TABLE_Y_MM = 2000.0;

bool ouvrirCamera(cv::VideoCapture& cap, int index = 0) {
    cap.open(index, cv::CAP_ANY);
    if (cap.isOpened()) return true;
    cap.open(index, cv::CAP_DSHOW);
    return cap.isOpened();
}

void dessinerGrilleEtHUD(cv::Mat& img, const ResultatDetection& res, int totalRefFeatures, bool afficherGrille) {
    int w = img.cols;
    int h = img.rows;

    auto mmVersPxX = [w](double x_mm) { return static_cast<int>(std::round((x_mm / TABLE_X_MM) * w)); };
    auto mmVersPxY = [h](double y_mm) { return static_cast<int>(std::round(h - (y_mm / TABLE_Y_MM) * h)); };

   
    if (afficherGrille) {
        const int pasMineurMM = 250; 
        const int pasMajeurMM = 500; 

        for (int x_mm = pasMineurMM; x_mm < static_cast<int>(TABLE_X_MM); x_mm += pasMineurMM) {
            int px = mmVersPxX(x_mm);
            bool majeur = (x_mm % pasMajeurMM == 0);
            cv::Scalar couleur = majeur ? cv::Scalar(120, 120, 120) : cv::Scalar(65, 65, 65);
            cv::line(img, cv::Point(px, 0), cv::Point(px, h), couleur, 1);
            if (majeur) {
                cv::putText(img, std::to_string(x_mm), cv::Point(px + 4, h - 10),
                            cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(220, 220, 220), 1);
            }
        }

        for (int y_mm = pasMineurMM; y_mm < static_cast<int>(TABLE_Y_MM); y_mm += pasMineurMM) {
            int py = mmVersPxY(y_mm);
            bool majeur = (y_mm % pasMajeurMM == 0);
            cv::Scalar couleur = majeur ? cv::Scalar(120, 120, 120) : cv::Scalar(65, 65, 65);
            cv::line(img, cv::Point(0, py), cv::Point(w, py), couleur, 1);
            if (majeur && py > 85) {
                cv::putText(img, std::to_string(y_mm), cv::Point(8, py - 5),
                            cv::FONT_HERSHEY_PLAIN, 0.9, cv::Scalar(220, 220, 220), 1);
            }
        }

        cv::circle(img, cv::Point(0, h - 1), 6, cv::Scalar(0, 255, 0), -1);
        cv::putText(img, "(0,0)", cv::Point(10, h - 10), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::line(img, cv::Point(w / 2 - 15, h / 2), cv::Point(w / 2 + 15, h / 2), cv::Scalar(255, 255, 0), 1);
        cv::line(img, cv::Point(w / 2, h / 2 - 15), cv::Point(w / 2, h / 2 + 15), cv::Scalar(255, 255, 0), 1);
    }

    double x_table_mm = (res.x / w) * TABLE_X_MM;
    double y_table_mm = ((h - res.y) / h) * TABLE_Y_MM; 
    double theta_table_deg = -res.thetaDeg;             

    if (res.carteTrouvee) {
        int px = static_cast<int>(std::round(res.x));
        int py = static_cast<int>(std::round(res.y));

        cv::line(img, cv::Point(px, 0), cv::Point(px, h), cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        cv::line(img, cv::Point(0, py), cv::Point(w, py), cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

        cv::line(img, cv::Point(px, py), cv::Point(px + 60, py), cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::ellipse(img, cv::Point(px, py), cv::Size(35, 35), 0.0, 0.0, res.thetaDeg, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        std::string labelCible = cv::format("X:%.0fmm Y:%.0fmm (%.1f deg)", x_table_mm, y_table_mm, theta_table_deg);
        cv::putText(img, labelCible, cv::Point(px + 12, py - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 2);
    }

    cv::Mat overlay = img.clone();
    cv::rectangle(overlay, cv::Point(0, 0), cv::Point(w, 80), cv::Scalar(20, 20, 20), -1);
    cv::addWeighted(overlay, 0.75, img, 0.25, 0, img);

    double ratioInliers = (res.nbMatchesBruts > 0)
                              ? (100.0 * res.nbFeaturesReconnues / res.nbMatchesBruts)
                              : 0.0;
    std::string alerteMouvement = (res.nbPointsMouvement > 30) ? " [MOUVEMENT]" : "";
    std::string txtFeatures = cv::format(
        "Inliers: %d/%d (RANSAC %.0f%%) | Pts Rouges (Mouv): %d%s",
        res.nbFeaturesReconnues, totalRefFeatures, ratioInliers, res.nbPointsMouvement, alerteMouvement.c_str()
    );
    cv::Scalar couleurLigne1 = (res.nbPointsMouvement > 30) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    cv::putText(img, txtFeatures, cv::Point(12, 26),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, couleurLigne1, 1, cv::LINE_AA);

    if (res.carteTrouvee) {
        std::string txtPose = cv::format(
            "TABLE (mm) -> X: %6.0f / 3000 | Y: %6.0f / 2000 | Theta: %+5.1f deg  [g] Grille [r] Ref",
            x_table_mm, y_table_mm, theta_table_deg
        );
        cv::putText(img, txtPose, cv::Point(12, 58),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    } else {
        std::string txtAttente = "TABLE (3000x2000) : Recherche de la carte...  [g] Grille | [r] Capturer Ref";
        cv::putText(img, txtAttente, cv::Point(12, 58),
                    cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(0, 165, 255), 1, cv::LINE_AA);
    }
}

int main() {
    std::filesystem::path dossierCourant = std::filesystem::current_path();
    std::string fichierYml = (dossierCourant / "carte_features.yml").string();

    DetecteurAKAZE detecteur(0.001f);
    bool cartePrete = detecteur.chargerCarteReference(fichierYml);

    cv::VideoCapture cap;
    if (!ouvrirCamera(cap, 0)) {
        std::cerr << "[ERREUR] Impossible d'ouvrir la camera !" << std::endl;
        return -1;
    }

    bool afficherGrille = true;
    cv::Mat frame, affichage;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        ResultatDetection res;
        if (cartePrete) {
            res = detecteur.analyserFrame(frame, affichage);
        } else {
            affichage = frame.clone();
            detecteur.detecterMouvement(frame, affichage, res, 25.0);
        }
        dessinerGrilleEtHUD(affichage, res, detecteur.getNbFeaturesCarte(), afficherGrille);

        cv::imshow("Robot Temps Reel - Verification Grille & Features AKAZE", affichage);

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27 || key == 'q') {
            break;
        } else if (key == 'g' || key == 'G') {
            afficherGrille = !afficherGrille; 
        } else if (key == 'r' || key == ' ') {
            cv::Mat visuRef;
            std::string cheminImg = (dossierCourant / "reference.png").string();
            cv::imwrite(cheminImg, frame);
            if (detecteur.genererCarteReference(frame, fichierYml, visuRef)) {
                cartePrete = true;
                std::cout << "[SUCCES] Nouvelle carte enregistree (" << detecteur.getNbFeaturesCarte() << " features)." << std::endl;
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}