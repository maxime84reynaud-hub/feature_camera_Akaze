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
    std::string cheminImage = (dossierCourant / "reference.png").string();
    std::string fichierYml = (dossierCourant / "carte_features.yml").string();

    DetecteurAKAZE detecteur(0.001f);
    cv::Mat imageRef = cv::imread(cheminImage);

    // Si une image existe déjà, on propose de la conserver ou d'appuyer sur 'c' pour recapturer
    if (!imageRef.empty()) {
        std::cout << "[OK] Image existante trouvee : " << cheminImage << std::endl;
        cv::imshow("Image actuelle - [c] pour recapturer | [Autre touche] pour valider", imageRef);
        char choix = static_cast<char>(cv::waitKey(0));
        cv::destroyAllWindows();
        if (choix == 'c' || choix == 'C') {
            imageRef.release(); // Vide l'image pour déclencher la capture webcam ci-dessous
        }
    }

    // Si aucune référence n'est chargée (ou si l'utilisateur a appuyé sur 'c'), on ouvre la webcam
    if (imageRef.empty()) {
        cv::VideoCapture cap;
        if (!ouvrirCamera(cap, 0)) {
            std::cerr << "[ERREUR] Webcam introuvable" << std::endl;
            return -1;
        }

        std::cout << "Appuyez sur [ESPACE] ou [r] sur la fenetre video pour capturer." << std::endl;
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
                    std::cout << "[SUCCES] Image enregistree sous : " << cheminImage << std::endl;
                } else {
                    std::cerr << "[ERREUR] Echec enregistrement" << std::endl;
                }
                break;
            } else if (key == 27 || key == 'q') {
                return 0;
            }
        }
        cap.release();
        cv::destroyAllWindows();
    }

    // Extraction des features et génération du fichier carte_features.yml
    cv::Mat imageVisu;
    if (detecteur.genererCarteReference(imageRef, fichierYml, imageVisu)) {
        std::cout << "[SUCCES] " << detecteur.getNbFeaturesCarte()
                  << " features sauvegardees dans : " << fichierYml << std::endl;
        cv::imshow("Features enregistrees (Appuyez sur une touche pour fermer)", imageVisu);
        cv::waitKey(0);
    } else {
        std::cerr << "[ERREUR] Impossible de generer le fichier YML (pas assez de features)." << std::endl;
    }

    return 0;
}