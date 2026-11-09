#include "features.h"
#include <iostream>
#include <filesystem>

// Fonction robuste pour ouvrir la webcam sous Windows et Linux
bool ouvrirCamera(cv::VideoCapture& cap, int index = 0) {
    cap.open(index, cv::CAP_ANY);
    if (cap.isOpened()) return true;
    cap.open(index, cv::CAP_DSHOW);
    return cap.isOpened();
}

int main() {
    std::filesystem::path dossierCourant = std::filesystem::current_path();
    std::string cheminImage = (dossierCourant / "reference.png").string();
    std::string fichierYml = (dossierCourant / "carte_features.yml").string();

    std::cout << "=== OUTIL DE PREPARATION DE LA CARTE ===" << std::endl;
    std::cout << "Dossier de travail : " << dossierCourant.string() << std::endl;

    DetecteurAKAZE detecteur(0.001f);
    cv::Mat imageRef = cv::imread(cheminImage);

    if (!imageRef.empty()) {
        std::cout << "[OK] Image existante trouvee : " << cheminImage << std::endl;
        std::cout << "Appuyez sur [c] sur la fenetre pour recapturer via webcam, ou toute autre touche pour valider." << std::endl;
    }

    // Si aucune image n'existe, on ouvre la caméra directement
    if (imageRef.empty()) {
        std::cout << "Aucune 'reference.png' trouvee. Ouverture de la webcam..." << std::endl;
        cv::VideoCapture cap;
        if (!ouvrirCamera(cap, 0)) {
            std::cerr << "[ERREUR] Impossible d'ouvrir la webcam (verifiez qu'aucune autre fenetre ne l'utilise)." << std::endl;
            return -1;
        }

        std::cout << "-> Placez votre carte devant la camera et appuyez sur [ESPACE] (ou 'r') pour capturer." << std::endl;
        cv::Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            cv::Mat visu = frame.clone();
            cv::putText(visu, "Appuyez sur [ESPACE] ou [r] pour capturer l'image",
                        cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2);
            cv::imshow("Capture Reference - Webcam", visu);

            char key = static_cast<char>(cv::waitKey(1));
            if (key == ' ' || key == 'r') {
                imageRef = frame.clone();
                if (cv::imwrite(cheminImage, imageRef)) {
                    std::cout << "[SUCCES] Image bien enregistree sous : " << cheminImage << std::endl;
                } else {
                    std::cerr << "[ERREUR] Echec de l'enregistrement de l'image !" << std::endl;
                }
                break;
            } else if (key == 27 || key == 'q') {
                return 0;
            }
        }
        cap.release();
        cv::destroyAllWindows();
    }

    // Extraction des features et génération du fichier YML
    cv::Mat imageVisu;
    if (detecteur.genererCarteReference(imageRef, fichierYml, imageVisu)) {
        std::cout << "[SUCCES] " << detecteur.getNbFeaturesCarte()
                  << " features extraites et sauvegardees dans : " << fichierYml << std::endl;
        cv::imshow("Features AKAZE enregistrees (Appuyez sur une touche pour fermer)", imageVisu);
        cv::waitKey(0);
    } else {
        std::cerr << "[ERREUR] Impossible de generer le fichier YML (pas assez de details sur l'image)." << std::endl;
    }

    return 0;
}