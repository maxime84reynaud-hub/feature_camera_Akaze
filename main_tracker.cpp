#include "features.h"
#include <iostream>
#include <filesystem>

bool ouvrirCamera(cv::VideoCapture& cap, int index = 0) {
    cap.open(index, cv::CAP_ANY);
    if (cap.isOpened()) return true;
    cap.open(index, cv::CAP_DSHOW);
    return cap.isOpened();
}

int main() {
    std::filesystem::path dossierCourant = std::filesystem::current_path();
    std::string fichierYml = (dossierCourant / "carte_features.yml").string();

    std::cout << "=== SUIVI TEMPS REEL ROBOT ===" << std::endl;
    std::cout << "Recherche de la carte dans : " << fichierYml << std::endl;

    DetecteurAKAZE detecteur(0.001f);
    bool cartePrete = detecteur.chargerCarteReference(fichierYml);

    if (cartePrete) {
        std::cout << "[OK] Carte chargee avec succes (" << detecteur.getNbFeaturesCarte() << " features)." << std::endl;
    } else {
        std::cout << "[ATTENTION] Fichier 'carte_features.yml' absent !" << std::endl;
        std::cout << "-> Vous pourrez appuyer sur [r] dans la fenetre camera pour capturer la carte en direct." << std::endl;
    }

    cv::VideoCapture cap;
    if (!ouvrirCamera(cap, 0)) {
        std::cerr << "[ERREUR] Impossible d'ouvrir la camera !" << std::endl;
        return -1;
    }

    std::cout << "Camera ouverte ! Commandes : [r] Definir nouvelle reference | [q / Echap] Quitter" << std::endl;

    cv::Mat frame, affichage;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        if (cartePrete) {
            ResultatDetection res = detecteur.analyserFrame(frame, affichage);
            if (res.carteTrouvee) {
                std::string infos = cv::format("X: %.1f | Y: %.1f | Theta: %.1f deg | Inliers: %d",
                                               res.x, res.y, res.thetaDeg, res.nbFeaturesReconnues);
                cv::putText(affichage, infos, cv::Point(15, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2);
            } else {
                cv::putText(affichage, "Recherche carte... (Inliers: " + std::to_string(res.nbFeaturesReconnues) + ") | [r] Recapturer",
                            cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 165, 255), 2);
            }
        } else {
            affichage = frame.clone();
            cv::putText(affichage, "Aucune carte : Appuyez sur [r] pour capturer la reference",
                        cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }

        cv::imshow("Robot Temps Reel - Reconnaissance AKAZE", affichage);

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27 || key == 'q') {
            break;
        } else if (key == 'r' || key == ' ') {
            cv::Mat visuRef;
            std::string cheminImg = (dossierCourant / "reference.png").string();
            cv::imwrite(cheminImg, frame);
            if (detecteur.genererCarteReference(frame, fichierYml, visuRef)) {
                cartePrete = true;
                std::cout << "[SUCCES] Nouvelle carte enregistree (" << detecteur.getNbFeaturesCarte()
                          << " features) dans : " << fichierYml << std::endl;
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}