# Cahier des charges des outils utilisés

Projet : détection de plantes malades (PlantDoc) embarquée sur un smartphone Android.
Ce document liste, pour chaque outil, son rôle, sa version, la raison de son choix et ses limites.

## 1. Besoin à couvrir
| Besoin | Exigence |
|---|---|
| Modèle | Un modèle de détection non trivial, entraîné ou récupéré, sur un dataset de choix |
| Embarqué | Le modèle tourne sur un dispositif embarqué (ici : smartphone Android simulé) |
| Optimisation | Une à trois optimisations comparées (quantization) |
| Mesures | Précision (mAP), taille du fichier, vitesse d'inférence, avant et après quantization |
| Déploiement | La meilleure variante quantifiée est déployée sur la cible |
| Reproductibilité | Les étapes doivent pouvoir être relancées depuis le dépôt |

## 2. Données
| Outil | Version | Rôle | Pourquoi |
|---|---|---|---|
| **PlantDoc** (détection) | 2020, CC BY 4.0 | Images de feuilles avec boîtes, 29 classes | Vraie détection avec boîtes, photos de terrain ; licence libre |
| **Git** | — | Récupération du dépôt PlantDoc | Source officielle du jeu de données |
| **Pillow** | 13.0.0 | Lecture et réduction des images (640 px) | Léger, suffit à la conversion |
| **pandas** | 3.1.1 | Lecture des étiquettes CSV, tableaux de résultats | Format CSV de PlantDoc, export des mesures |

Limites : sous Windows, des noms de fichiers trop longs ont empêché l'extraction d'environ 93 images d'entraînement et 5 de test.

## 3. Entraînement
| Outil | Version | Rôle | Pourquoi |
|---|---|---|---|
| **Python** | 3.13.7 | Langage de tous les scripts | Écosystème standard de l'IA |
| **PyTorch** | 2.15.0 | Moteur d'apprentissage | Requis par Ultralytics |
| **Ultralytics (YOLOv8n)** | 8.4.207 | Fine-tuning et export du détecteur | Modèle léger (3 M de paramètres), poids COCO pré-entraînés, export ONNX intégré |
| **NumPy** | 2.5.4 | Calculs de pré/post-traitement et de mAP | Standard |

Choix du modèle : YOLOv8n (version « nano »), assez petit pour un CPU et un téléphone. Entraînement sur **CPU** (pas de GPU) : 416 px, 25 epochs au maximum, arrêt anticipé. C'est ce qui limite le mAP atteint.

## 4. Conversion et quantization
| Outil | Version | Rôle | Pourquoi |
|---|---|---|---|
| **ONNX** | 1.23.2 | Format d'échange du modèle | Fonctionne sur PC et sur Android, sans PyTorch |
| **onnxslim** | 0.1.99 | Simplification du graphe à l'export | Utilisé par Ultralytics |
| **ONNX Runtime (PC)** | 1.31.1 | Exécution des modèles et quantization INT8 | Quantization dynamique et statique (QDQ) intégrées, même moteur que sur Android |
| **onnxconverter-common** | 1.16.0 | Conversion FP32 → FP16 | Conversion simple du graphe ONNX |
| **OpenCV** | 5.0.0.93 | Suppression des doublons de boîtes (NMS) | Implémentation rapide |

Variantes produites : FP32 (référence), FP16, INT8 dynamique, INT8 statique (par canal, calibration sur 100 images de train). La tête de détection est gardée en FP32.

Choix justifié : la quantization **post-training** (sans réentraînement) est rapide sur CPU. Le **QAT** (apprentissage conscient de la quantization) n'a pas été tenté.

## 5. Évaluation
| Élément | Détail |
|---|---|
| Métriques | mAP50, mAP50-95 (calcul maison, `scripts/plantdoc_pipeline.py`), taille du fichier, latence médiane |
| Jeu de mesure | Test officiel PlantDoc (231 images), séparé de la validation |
| Matériel | CPU du PC (Intel Core i5-12400F), un seul passage, PC non isolé |
| Sorties | `results/benchmark_plantdoc.csv` |

Limites : jeu de test petit et classes déséquilibrées ; écarts de l'ordre du point de mAP non significatifs ; latences à comparer seulement entre variantes.

## 6. Application et cible embarquée
| Outil | Version | Rôle | Pourquoi |
|---|---|---|---|
| **Android Studio** | Build 262.10968 (JDK intégré OpenJDK 25) | Environnement de développement et émulateur | Outil officiel Android |
| **Émulateur Android** | Pixel 6, API 34 (Android 14), x86_64 | Cible simulée | Pas de téléphone physique disponible ; x86_64 pour profiter de la virtualisation |
| **Gradle** | 9.6.0 | Construction de l'APK | Standard Android |
| **Android Gradle Plugin** | 8.5.2 | Compilation Android | Compatible avec le projet |
| **Kotlin** | 1.9.24 (cible JVM 17) | Langage de l'application | Langage recommandé pour Android |
| **ONNX Runtime Android** | 1.19.2 | Exécution du modèle sur le téléphone | Même format ONNX qu'à l'export, pas de conversion supplémentaire |
| SDK Android | compileSdk 34, minSdk 26 | Compatibilité | Android 8.0 et plus |

L'application embarque les 4 variantes du modèle (environ 26 Mo), les noms de classes (`classes.txt`) et propose : choix de photo, boîtes et verdict, boutons de version, benchmark de latence.

Limites : l'émulateur tourne sur le CPU du PC, donc ses latences ne représentent pas un téléphone ARM réel ; l'orientation EXIF n'est pas gérée.

## 7. Outils de travail
| Outil | Rôle |
|---|---|
| **Claude Code** (assistant IA) | Aide à l'écriture des scripts, de l'application et de la documentation |
| **Windows 11** | Système de développement |
| Éditeur de code et terminal | Édition des scripts et lancement des commandes |

## 8. Contraintes et exigences non fonctionnelles
| Contrainte | Choix |
|---|---|
| Pas de GPU | Modèle nano, images réduites à 416 px, peu d'epochs |
| Fonctionnement hors réseau | Inférence entièrement locale sur le téléphone |
| Licences | Données CC BY 4.0 ; modèle YOLOv8 (Ultralytics, licence AGPL-3.0, à respecter pour toute diffusion) |
| Reproductibilité | Scripts versionnés dans `scripts/`, résultats dans `results/`, étapes décrites dans `README.md` |
| Vie privée | Aucune donnée n'est envoyée : les photos restent sur l'appareil |
