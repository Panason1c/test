# PlantDoc Mobile : détection de plantes malades embarquée sur Android

TP noté « Embarquer un modèle » (niveau 2 : packaging, quantization comparée, déploiement sur la cible).

Un détecteur YOLOv8n fine-tuné sur **PlantDoc** repère les feuilles saines ou malades sur une photo. Il tourne **entièrement sur le téléphone** (sans réseau) et existe en quatre versions (FP32, FP16, INT8 dynamique, INT8 statique) que l'application permet de comparer.

| | |
|---|---|
| **Modèle** | YOLOv8n (poids COCO) fine-tuné, entrée 416 px, 29 classes |
| **Dataset** | [PlantDoc](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset), CC BY 4.0 |
| **Optimisations** | FP16, INT8 dynamique, INT8 statique (post-training) |
| **Cible** | Smartphone Android, simulé avec l'émulateur d'Android Studio (ONNX Runtime) |
| **Résultats et analyse** | [notebooks/EDA.ipynb](notebooks/EDA.ipynb) ; export déjà généré : [results/EDA.html](results/EDA.html) |
| **Outils utilisés** | [CAHIER_DES_CHARGES_OUTILS.md](CAHIER_DES_CHARGES_OUTILS.md) |

## Installation complète (de zéro)

Toutes les étapes pour installer et lancer le projet sur un PC Windows, de l'environnement Python jusqu'à l'application sur l'émulateur. Les commandes sont pour **PowerShell**, à lancer depuis la racine du projet (le dossier qui contient ce `README.md`).

| Étape | Durée approximative |
|---|---|
| 1 à 3 : outils, projet, environnement Python | 15 à 30 min |
| 4 : données PlantDoc | 5 à 10 min |
| 5 : entraînement (CPU) | environ 1 h pour 25 epochs, environ 2 h 30 pour 60 |
| 6 : quantization et benchmark | 2 à 5 min |
| 7 : analyse exploratoire (facultatif) | 2 min |
| 8 à 9 : Android Studio, émulateur, application | 30 à 60 min (téléchargements) |

> **Raccourci** : le dossier `models/` contient déjà les quatre modèles et `android/app/src/main/assets/models/` les embarque. Pour seulement lancer l'application, passe directement aux étapes **8 et 9**.

### 1. Installer les outils (une seule fois)
| Outil | Version testée | Où le trouver |
|---|---|---|
| Python | 3.13 (3.10 et plus devrait convenir) | https://www.python.org/downloads/ (cocher « Add Python to PATH ») |
| Git | récent | https://git-scm.com/download/win |
| Android Studio | récent (build 262) | https://developer.android.com/studio |

Espace disque conseillé : environ 15 Go (Python et dépendances 3 Go, données 1,2 Go, Android Studio et SDK 8 Go, émulateur 3 Go). Mémoire : 16 Go de RAM recommandés. Vérifier les installations :

```powershell
python --version
git --version
```

### 2. Récupérer le projet
Copier le dossier du projet sur le PC (par exemple sur le Bureau), puis ouvrir PowerShell dedans :

```powershell
cd "C:\Users\<toi>\Desktop\TP MLOPS"
```

### 3. Créer l'environnement Python
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Si PowerShell refuse d'activer l'environnement (« l'exécution de scripts est désactivée »), lancer une fois `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, ou utiliser `.venv\Scripts\python` à la place de `python` dans les commandes suivantes. L'installation télécharge PyTorch (plusieurs centaines de Mo). Vérifier :

```powershell
python -c "import ultralytics, onnxruntime, cv2; print('OK', ultralytics.__version__)"
```

### 4. Récupérer et convertir PlantDoc
```powershell
# Récupérer le jeu de données (environ 1 Go)
git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset.git data/plantdoc_raw/PlantDoc-Object-Detection-Dataset
# Convertir au format YOLO -> data/plantdoc_yolo/ (train, val, test, 29 classes)
python scripts/prep_plantdoc.py data/plantdoc_raw/PlantDoc-Object-Detection-Dataset
```
Sous Windows, `git clone` affiche des erreurs `Filename too long` : quelques images aux noms trop longs ne sont pas extraites. Le script de conversion les ignore, c'est normal. Pour tenter de les récupérer, activer les chemins longs avant le clonage (`git config --global core.longpaths true`, et la stratégie Windows « Activer les chemins d'accès Win32 longs ») ; ce contournement n'a pas été testé. Supprimer ensuite le dossier caché `data/plantdoc_raw/PlantDoc-Object-Detection-Dataset/.git` libère environ 900 Mo.

### 5. Entraîner le modèle (facultatif si `models/` est déjà rempli)
```powershell
python scripts/finetune_plantdoc.py        # 25 epochs, environ 1 h
python scripts/finetune_plantdoc.py 60     # ou 60 epochs, environ 2 h 30
```
L'entraînement tourne sur CPU et télécharge au premier lancement les poids de départ `yolov8n.pt` (6,5 Mo). Il exporte le meilleur modèle en `models/plantdoc_fp32.onnx` ; les poids d'origine sont dans `runs/detect/runs/plantdoc/`. Laisser le PC branché et ne pas le mettre en veille.

### 6. Quantizer et mesurer
```powershell
python scripts/plantdoc_pipeline.py
```
Crée `models/plantdoc_fp16.onnx`, `plantdoc_int8_dynamic.onnx` et `plantdoc_int8_static.onnx`, mesure précision, taille et latence de chaque variante sur le jeu de test, et écrit `results/benchmark_plantdoc.csv`.

### 7. Analyse exploratoire (facultatif)
C'est ici que se trouvent les chiffres, les tableaux, les graphiques et les performances. L'analyse est le notebook `notebooks/EDA.ipynb`, à exécuter cellule par cellule :
```powershell
pip install jupyter
jupyter notebook notebooks/EDA.ipynb
```
Ou l'ouvrir directement dans VS Code avec l'environnement `.venv`. Puis *Run All* (1 à 2 minutes). Les graphiques s'affichent dans le notebook, rien n'est écrit sur le disque.

### 8. Installer l'application Android
1. **Mettre les modèles dans l'app** (nécessaire seulement si tu as ré-entraîné ou re-quantizé aux étapes 5 ou 6) :
   ```powershell
   Copy-Item models\plantdoc_*.onnx android\app\src\main\assets\models\ -Force
   ```
2. **Installer Android Studio** : lancer l'installeur, choisir l'installation **Standard**, laisser télécharger le SDK Android (environ 3 Go).
3. **Activer la virtualisation** : si l'émulateur le demande, ouvrir « Activer ou désactiver des fonctionnalités Windows », cocher **Plateforme de l'hyperviseur Windows**, redémarrer le PC. Vérifier aussi que la virtualisation est activée dans le BIOS.
4. **Ouvrir le projet** : écran d'accueil, **Open**, choisir le dossier `android` (celui qui contient `settings.gradle.kts`, pas la racine du projet), **Trust Project**. Attendre la fin de « Gradle sync » (plusieurs minutes, internet requis). Aucune configuration de JDK n'est nécessaire : Android Studio utilise le sien.
5. **Créer le téléphone virtuel** : *Tools > Device Manager > Create Device*, **Phone > Pixel 6**, image système **API 34 (Android 14) x86_64** (*Download* si besoin), *Finish*.

### 9. Lancer l'application et tester
1. Dans la barre du haut d'Android Studio, choisir **app** et le **Pixel 6**, puis cliquer sur le triangle vert **Run**. Le premier démarrage de l'émulateur prend 1 à 3 minutes, puis l'app **PlantDoc** s'ouvre (le premier chargement des modèles peut être lent).
2. Glisser des photos de `exemples_test/` sur la fenêtre de l'émulateur : elles arrivent dans les Téléchargements.
3. **Choisir une photo de feuille** : l'app affiche les boîtes, le verdict et le temps d'inférence. Comparer le verdict au nom du fichier.
4. Cliquer sur les 4 boutons de version (FP32, FP16, INT8 dyn., INT8 stat.) : la photo est ré-analysée avec la nouvelle version.
5. **Benchmark de latence** : le lancer pour chaque version et noter les valeurs.
6. **Capture d'écran** : icône appareil photo de la barre latérale de l'émulateur.

### Vérifier que tout fonctionne
| Contrôle | Résultat attendu |
|---|---|
| `python scripts/prep_plantdoc.py ...` | trois lignes `train`, `val`, `test` avec un nombre d'images, puis `29 classes` |
| `python scripts/plantdoc_pipeline.py` | une ligne de résultats par variante (fp32, fp16, int8_dynamic, int8_static), puis `results/benchmark_plantdoc.csv` créé |
| App sur l'émulateur | une boîte et un verdict sur une feuille de `exemples_test/`, sans message d'erreur |

## Contenu du projet

```
README.md                      ce guide d'installation
CAHIER_DES_CHARGES_OUTILS.md   outils utilisés, versions, rôle
requirements.txt               dépendances Python (versions testées)
scripts/
  prep_plantdoc.py             conversion PlantDoc (CSV) -> format YOLO
  finetune_plantdoc.py         fine-tuning + export ONNX FP32
  plantdoc_pipeline.py         quantization, mesures, results/benchmark_plantdoc.csv
notebooks/
  EDA.ipynb                    analyse exploratoire (données, entraînement, performances)
models/                        plantdoc_{fp32,fp16,int8_dynamic,int8_static}.onnx
results/                       benchmark_plantdoc.csv, EDA.html, EDA.md, eda/
android/                       application Android (Kotlin, ONNX Runtime)
exemples_test/                 10 photos de feuilles du jeu de test, nommées d'après la classe attendue
data/                          données PlantDoc brutes et converties (créées à l'étape 4)
runs/                          poids d'entraînement (créés à l'étape 5)
```

## Application Android

L'app choisit une photo, dessine les boîtes (**vert** = feuille saine, **rouge** = malade) avec le nom de la classe et la confiance, puis affiche un verdict : « Maladie probable : … », « Feuille saine : … » ou « Aucune feuille détectée ». Sous 50 % de confiance, elle ajoute un avertissement. Quatre boutons changent la version du modèle (la dernière photo est ré-analysée) et un bouton lance le **benchmark de latence**. Version par défaut : INT8 statique.

## Problèmes fréquents
| Symptôme | Solution |
|---|---|
| `ModuleNotFoundError` | Lancer depuis la racine du projet, avec l'environnement `.venv` activé. |
| `git clone` : « Filename too long » | Normal sous Windows, ces images sont ignorées (voir étape 4). |
| `Activate.ps1` refusé | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (étape 3). |
| « Gradle sync failed » | Vérifier internet, puis *File > Sync Project with Gradle Files*. |
| `Inconsistent JVM-target compatibility` | Déjà corrigé dans `android/app/build.gradle.kts` (cible 17). |
| L'émulateur ne démarre pas | Activer la virtualisation dans le BIOS et l'hyperviseur Windows. |
| L'app plante | Onglet *Logcat* d'Android Studio, filtrer sur `AndroidRuntime`, lire l'erreur. |
| Latences élevées ou très variables | Normal sur émulateur : utiliser le bouton Benchmark (moyenne sur 30 passes), pas le temps d'une seule photo. |
| Erreur `Resize_output_cast0` (FP16) | Déjà corrigée dans `plantdoc_pipeline.py`. |
| Précision INT8 nulle | La tête de détection doit rester en FP32 (`nodes_to_exclude` dans `plantdoc_pipeline.py`). |
