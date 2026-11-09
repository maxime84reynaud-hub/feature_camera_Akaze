#include "features.h"
#include <iostream>
#include <filesystem>

bool ouvrirCamera(cv::VideoCapture& cap, int index = 0) {
    cap.open(index, cv::CAP_ANY);
    if (cap.isOpened()) return true;
    cap.open(index, cv::CAP_DSHOW);
    return cap.isOpened();
}

// Dessine la grille de coordonnées graduée, les compteurs de features et les alertes de mouvement
void dessinerGrilleEtHUD(cv::Mat& img, const ResultatDetection& res, int totalRefFeatures, bool afficherGrille) {
    int w = img.cols;
    int h = img.rows;

    // 1. Tracé de la grille graduée en pixels
    if (afficherGrille) {
        const int pasMineur = 50;
        const int pasMajeur = 100;

        // Lignes verticales (Axe X)
        for (int x = pasMineur; x < w; x += pasMineur) {
            bool majeur = (x % pasMajeur == 0);
            cv::Scalar couleur = majeur ? cv::Scalar(110, 110, 110) : cv::Scalar(65, 65, 65);
            cv::line(img, cv::Point(x, 0), cv::Point(x, h), couleur, 1);
            if (majeur) {
                cv::putText(img, std::to_string(x), cv::Point(x + 3, h - 8),
                            cv::FONT_HERSHEY_PLAIN, 0.85, cv::Scalar(200, 200, 200), 1);
            }
        }

        // Lignes horizontales (Axe Y)
        for (int y = pasMineur; y < h; y += pasMineur) {
            bool majeur = (y % pasMajeur == 0);
            cv::Scalar couleur = majeur ? cv::Scalar(110, 110, 110) : cv::Scalar(65, 65, 65);
            cv::line(img, cv::Point(0, y), cv::Point(w, y), couleur, 1);
            if (majeur && y > 85) { // Évite d'écrire sous le bandeau HUD du haut
                cv::putText(img, std::to_string(y), cv::Point(5, y - 4),
                            cv::FONT_HERSHEY_PLAIN, 0.85, cv::Scalar(200, 200, 200), 1);
            }
        }

        // Réticule central de la caméra (centre optique théorique)
        cv::line(img, cv::Point(w / 2 - 15, h / 2), cv::Point(w / 2 + 15, h / 2), cv::Scalar(255, 255, 0), 1);
        cv::line(img, cv::Point(w / 2, h / 2 - 15), cv::Point(w / 2, h / 2 + 15), cv::Scalar(255, 255, 0), 1);
    }

    // 2. Si la carte est trouvée : lignes de projection (X, Y) et rapporteur d'angle Theta
    if (res.carteTrouvee) {
        int px = static_cast<int>(std::round(res.x));
        int py = static_cast<int>(std::round(res.y));

        // Lignes de visée reliant la cible aux graduations X et Y
        cv::line(img, cv::Point(px, 0), cv::Point(px, h), cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        cv::line(img, cv::Point(0, py), cv::Point(w, py), cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

        // Axe de référence 0 degré (horizontal blanc) et arc de l'angle Theta
        cv::line(img, cv::Point(px, py), cv::Point(px + 60, py), cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::ellipse(img, cv::Point(px, py), cv::Size(35, 35), 0.0, 0.0, res.thetaDeg, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        // Étiquette directe à côté du centre détecté
        std::string labelCible = cv::format("(%.0f, %.0f) %.1f deg", res.x, res.y, res.thetaDeg);
        cv::putText(img, labelCible, cv::Point(px + 12, py - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 2);
    }

    // 3. Bandeau HUD semi-transparent en haut de l'écran
    cv::Mat overlay = img.clone();
    cv::rectangle(overlay, cv::Point(0, 0), cv::Point(w, 80), cv::Scalar(20, 20, 20), -1);
    cv::addWeighted(overlay, 0.75, img, 0.25, 0, img);

    // Ligne 1 : Compteur de features en direct + Points rouges de mouvement
    double ratioInliers = (res.nbMatchesBruts > 0)
                              ? (100.0 * res.nbFeaturesReconnues / res.nbMatchesBruts)
                              : 0.0;
    std::string alerteMouvement = (res.nbPointsMouvement > 30) ? " [OBSTACLE EN MOUVEMENT]" : "";
    std::string txtFeatures = cv::format(
        "Inliers: %d/%d (RANSAC %.0f%%) | Pts Rouges (Mouv): %d%s",
        res.nbFeaturesReconnues, totalRefFeatures, ratioInliers, res.nbPointsMouvement, alerteMouvement.c_str()
    );
    cv::Scalar couleurLigne1 = (res.nbPointsMouvement > 30) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    cv::putText(img, txtFeatures, cv::Point(12, 26),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, couleurLigne1, 1, cv::LINE_AA);

    // Ligne 2 : Coordonnées X, Y, Theta et commandes
    if (res.carteTrouvee) {
        std::string txtPose = cv::format(
            "POSE -> X: %5.1f px | Y: %5.1f px | Theta: %+5.1f deg   [g] Grille [r] Ref",
            res.x, res.y, res.thetaDeg
        );
        cv::putText(img, txtPose, cv::Point(12, 58),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    } else {
        std::string txtAttente = "POSE : Recherche de la carte (min 8 inliers)...  [g] Grille | [r] Capturer Ref";
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

        // Superposition de la grille graduée, de la pose et des compteurs en direct
        dessinerGrilleEtHUD(affichage, res, detecteur.getNbFeaturesCarte(), afficherGrille);

        cv::imshow("Robot Temps Reel - Verification Grille & Features AKAZE", affichage);

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27 || key == 'q') {
            break;
        } else if (key == 'g' || key == 'G') {
            afficherGrille = !afficherGrille; // Permet d'activer/masquer la grille avec 'g'
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