# 5A-INFO BigData : API de serving pour un modèle prédictif en ML, avec application web utilisateur, système de reporting et de monitoring en temps réel

## CLI utiles pour le projet
Dans l'ensemble du projet, l'utilisation des CLI associées à des conteneurs Docker permettent une utilisation facile des différents modules. Leur image et configuration sont décrites dans les fichiers DockerFile et docker-compose.yml respectifs, où sont spécifiés l'exécution des fichiers de requirements permettant l'installation de tous les packages python nécessaires.
Voici les différentes CLI Docker à utiliser selon les modules du projet depuis le dossier ML_project  :
- partie serving : **docker compose -f serving/docker-compose.yml up**
- partie webapp : **docker compose -f webapp/docker-compose.yml up**
- partie reporting : **docker compose -f reporting/docker-compose.yml up**
- partie monitoring : **docker compose -f monitoring/docker-compose.yml up**
    >Si Grafana indique dans le terminal une erreur, celle-ci sera sans doute liée à une erreur de lecture d'un fichier. Ainsi il sera sans doute utile de saisir la CLI suivante : **sudo chmod -R 777 ./data**.
    Par ailleurs lors de la première connexion sur Grafana, il est demandé le nom d'utilisateur et le mdp: par défaut il faut saisir *admin* pour les 2 champs.


## Outils technologiques utilisés :
- Python (Scikit-Learn, Fast-API, Streamlit, Evidently)
- Conteneurs Docker
- Prometheus et Grafana

## Dataset choisi : 
[Prédiction sur la mort (1) ou non (0) d'une personne admise à l'hôpital](https://www.kaggle.com/datasets/mitishaagarwal/patient/data)

## Composition du groupe : 
-   BLUMET Thomas (Double-diplôme)
-   BURDAIRON Florian
-   GHAZEL Hassen
-   HALVICK Thomas (Double-diplôme)
-   MATTAR Omar

# 1 ) Préambule : description du dataset

## Variables catégorielle :

| Nom de la variable     | Description                                              | Valeurs possibles                                                                 |
|------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------|
| `ethnicity`            | Ethnicité ou appartenance culturelle du patient.         | Caucasian, African American, Other/Unknown, Hispanic, Asian                       |
| `gender`               | Sexe du patient.                                         | M, F                                                                              |
| `icu_admit_source`     | Source d'admission avant l'entrée en soins intensifs.    | Accident & Emergency, Operating Room / Recovery, Floor, Other Hospital, Other ICU |
| `icu_stay_type`        | Type de séjour en soins intensifs.                       | admit, transfer, readmit                                                          |
| `icu_type`             | Type d'unité de soins intensifs où le patient est admis. | Med-Surg ICU, MICU, Neuro ICU, CCU-CTICU, SICU                                    |
| `apache_3j_bodysystem` | Système corporel affecté selon APACHE III-J.             | Cardiovascular, Neurological, Sepsis, Respiratory, Gastrointestinal               |
| `apache_2_bodysystem`  | Système corporel affecté selon APACHE II.                | Cardiovascular, Neurologic, Respiratory, Gastrointestinal, Metabolic              |

## Ensembles des variables du dataset

| Nom de la variable              | Description                                               | Valeurs possibles (- indique la présence de valeurs numériques entières ou flottantes) |
|---------------------------------|-----------------------------------------------------------|----------------------------------------------------------------------------------------|
| `encounter_id`                  | Identifiant unique du séjour en unité.                    | -                                                                                      |
| `patient_id`                    | Identifiant unique du patient.                            | -                                                                                      |
| `hospital_id`                   | Identifiant unique de l'hôpital.                          | -                                                                                      |
| `age`                           | Âge du patient à l'admission en unité.                    | -                                                                                      |
| `bmi`                           | Indice de masse corporelle du patient.                    | -                                                                                      |
| `elective_surgery`              | Admission pour chirurgie programmée.                      | 0 (Non), 1 (Oui)                                                                       |
| `ethnicity`                     | Ethnicité ou appartenance culturelle du patient.          | Caucasian, African American, Other/Unknown, Hispanic, Asian                            |
| `gender`                        | Sexe du patient.                                          | M, F                                                                                   |
| `height`                        | Taille du patient.                                        | -                                                                                      |
| `icu_admit_source`              | Source d'admission avant l'entrée en soins intensifs.     | Accident & Emergency, Operating Room / Recovery, Floor, Other Hospital, Other ICU      |
| `icu_id`                        | Identifiant unique de l’unité de soins intensifs.         | -                                                                                      |
| `icu_stay_type`                 | Type de séjour en soins intensifs.                        | admit, transfer, readmit                                                               |
| `icu_type`                      | Type d'unité de soins intensifs où le patient est admis.  | Med-Surg ICU, MICU, Neuro ICU, CCU-CTICU, SICU                                         |
| `pre_icu_los_days`              | Durée du séjour avant admission en soins intensifs.       | -                                                                                      |
| `weight`                        | Poids du patient.                                         | -                                                                                      |
| `apache_2_diagnosis`            | Code de diagnostic APACHE II.                             | -                                                                                      |
| `apache_3j_diagnosis`           | Code de diagnostic APACHE III-J.                          | -                                                                                      |
| `apache_post_operative`         | Statut post-opératoire APACHE.                            | 0 (Non), 1 (Oui)                                                                       |
| `arf_apache`                    | Présence d'insuffisance rénale aiguë.                     | 0 (Non), 1 (Oui)                                                                       |
| `gcs_eyes_apache`               | Score APACHE de l’ouverture des yeux (GCS).               | 1-4                                                                                    |
| `gcs_motor_apache`              | Score APACHE de la réponse motrice (GCS).                 | 1-6                                                                                    |
| `gcs_unable_apache`             | Évaluation GCS impossible (ex: sédation).                 | 0 (Non), 1 (Oui)                                                                       |
| `gcs_verbal_apache`             | Score APACHE de la réponse verbale (GCS).                 | 1-5                                                                                    |
| `intubated_apache`              | Patient intubé lors des gaz du sang les plus sévères.     | 0 (Non), 1 (Oui)                                                                       |
| `ventilated_apache`             | Patient ventilé mécaniquement.                            | 0 (Non), 1 (Oui)                                                                       |
| `d1_diasbp_max`                 | Pression artérielle diastolique max, jour 1.              | -                                                                                      |
| `d1_diasbp_min`                 | Pression artérielle diastolique min, jour 1.              | -                                                                                      |
| `d1_diasbp_mean`                | Pression artérielle diastolique moyenne, jour 1.          | -                                                                                      |
| `d1_heartrate_max`              | Fréquence cardiaque max, jour 1.                          | -                                                                                      |
| `d1_heartrate_min`              | Fréquence cardiaque min, jour 1.                          | -                                                                                      |
| `d1_heartrate_mean`             | Fréquence cardiaque moyenne, jour 1.                      | -                                                                                      |
| `d1_resprate_max`               | Fréquence respiratoire max, jour 1.                       | -                                                                                      |
| `d1_resprate_min`               | Fréquence respiratoire min, jour 1.                       | -                                                                                      |
| `d1_resprate_mean`              | Fréquence respiratoire moyenne, jour 1.                   | -                                                                                      |
| `d1_spo2_max`                   | Saturation en oxygène max, jour 1.                        | -                                                                                      |
| `d1_spo2_min`                   | Saturation en oxygène min, jour 1.                        | -                                                                                      |
| `d1_spo2_mean`                  | Saturation en oxygène moyenne, jour 1.                    | -                                                                                      |
| `d1_sysbp_max`                  | Pression artérielle systolique max, jour 1.               | -                                                                                      |
| `d1_sysbp_min`                  | Pression artérielle systolique min, jour 1.               | -                                                                                      |
| `d1_sysbp_mean`                 | Pression artérielle systolique moyenne, jour 1.           | -                                                                                      |
| `apache_4a_hospital_death_prob` | Probabilité de décès à l’hôpital selon APACHE IVa.        | -                                                                                      |
| `apache_4a_icu_death_prob`      | Probabilité de décès en soins intensifs selon APACHE IVa. | -                                                                                      |
| `aids`                          | Patient atteint du SIDA.                                  | 0 (Non), 1 (Oui)                                                                       |
| `cirrhosis`                     | Présence d'une cirrhose.                                  | 0 (Non), 1 (Oui)                                                                       |
| `diabetes_mellitus`             | Diabète nécessitant un traitement.                        | 0 (Non), 1 (Oui)                                                                       |
| `hepatic_failure`               | Insuffisance hépatique sévère.                            | 0 (Non), 1 (Oui)                                                                       |
| `immunosuppression`             | Patient immunodéprimé avant admission.                    | 0 (Non), 1 (Oui)                                                                       |
| `leukemia`                      | Présence d'une leucémie.                                  | 0 (Non), 1 (Oui)                                                                       |
| `lymphoma`                      | Présence d'un lymphome.                                   | 0 (Non), 1 (Oui)                                                                       |
| `solid_tumor_with_metastasis`   | Cancer avec métastases.                                   | 0 (Non), 1 (Oui)                                                                       |
| `apache_3j_bodysystem`          | Système corporel affecté selon APACHE III-J.              | Cardiovascular, Neurological, Sepsis, Respiratory, Gastrointestinal                    |
| `apache_2_bodysystem`           | Système corporel affecté selon APACHE II.                 | Cardiovascular, Neurologic, Respiratory, Gastrointestinal, Metabolic                   |
| `hospital_death`                | Statut du patient à la sortie de l'hôpital.               | 0 (Survivant), 1 (Décédé)                                                              |

# 2) Description du projet
## Étape 1 : prétraitement des données et choix du modèle de classification
Dans un [fichier notebook](/scripts/notebook.ipynb), nous avons réalisé un premier traitement sur les données, afin de savoir notamment s'il existait des variables manquantes et catégorielles, combien le dataset en contenait, et de pouvoir les imputer et encoder afin de créer le fichier ref_data.csv qui permettra ensuite de faire un réentraînement du modèle. Dans notre cas, nous avons aussi enlevé les colonnes `encounter_id` et `patient_id` qui ne servent pas à la prédiction, de même que la colonne vide précédant `hospital_death` dans le fichier [init_data.csv](data/init_data.csv) qui est le fichier de base téléchargé depuis Kaggle.
Cela a permis aussi de choisir quelle fonction de scoring nous allions utiliser. En effet ce choix est crucial car il détermine a lui seul la sélection du meilleur modèle de classification et sa performance, ainsi que la sélection du nombre de variables pertinentes pour l'entraînement du modèle ainsi que le choix de ses hyperparamètres. 

**Dans notre contexte d'étude, il s'avère important de limiter le nombre de faux négatifs (FN), c'est-à-dire le nombre de personnes qui, d'après la prédiction du modèle, ne vont pas mourir, alors qu'en réalité cela va être le cas. De fait, la fonction de scoring implémentée se base sur la notion de rappel (Recall), avec la formule suivante** :
> (Recall(target,prediction,classe 0) + Recall(target,prediction,classe 1))/2

Cette formule permet de tenir compte aussi bien des faux négatifs que des faux positifs.

Après avoir sélectionné la fonction de scoring il s'est ensuivi l'exécution de la sélection du meilleur modèle de classification selon une comparaison croisée (méthode des K-Folds) partitionnant en K=10 le dataset initial. En effet, cela permet de répartir équitablement les données afin que toutes puissent servir de données notamment d'entraînement, afin que le modèle de classification soit le moins biaisé possible. 
Cette comparaison croisée est effectuée via 3 stratégies :
- en utilisant les données tel qu'imputées précédemment (**stratégie 'natural'**)
- en les ayant normalisé (via la classe *StandardScaler* de scikit) (**stratégie 'normalized'**)
- en ayant appliqué l'ACP (Analyse en Composantes Principales) impliquant de les avoir normalisé au préalable (**stratégie 'pca'**)

Le meilleur modèle choisi (avec des paramètres très basique) est retourné avec la stratégie à appliquer sur les données. Ce sont ces données qui sont sauvegardées dans le fichier [ref_data.csv](/data/ref_data.csv).
S'ensuit alors le classement des variables du dataset selon leur importance, et la sélection des variables les plus pertinentes qui serviront à entraîner le modèle choisi. Mais avant de passer à l'entraînement de ce dernier, il s'avère nécessaire de sélectionner les meilleurs hyperparamètres afin d'obtenir une performance de classification accrue, ceci grâce à l'utilisation de la fonction de *GridSearchCV* de scikit sur un ensemble d'hyperparamètres donné pour le modèle choisi.

>Pour notre cas d'étude, **le meilleur modèle choisi a été le modèle *GaussianNB* avec une stratégie naturelle de prétraitement des données**.

## Étape 2 : mise en production du modèle avec des Pipelines
Une fois la sélection du modèle réalisé ainsi que son entraînement, le point vital du projet a été de le mettre en production via des pipelines (classe *Pipeline* de scikit). Pour notre cas, nous avons enregistré [2 pipelines](/artifacts) qui sont chargés lors de l'utilisation de l'API de serving :
- le pipeline **imputer.pkl** lié à l'imputation des données manquantes et l'encodage (via la classe [*OrdinalEncoder*](scripts/utils.py#L572)) des variables catégorielles
- le pipeline **model.pkl** lié au modèle de classification utilisé pour faire les prédictions

## Étape 3 : Élaboration de l'API de serving avec FastAPI
Afin de pouvoir prédire la mort ou non de nouveaux patients entrant dans un hôpital via l'interface web utilisateur, l'utilisation du pipeline de model nécessite de passer par [un endpoint de prédiction (/predict)](serving/api.py#L107). Ceci a été rendu possible par l'utilisation de l'import de Fast-API. L'endpoint prend ainsi en paramètre un objet au format JSON qui est retransformé afin de pouvoir être utilisé comme donnée à prédire par la méthode [*predit()*](serving/api.py#L130).

## Étape 4 : Création d'une webapp utilisateur avec Streamlit
Pour permettre une utilisation de l'API par un utilisateur lambda, une webapp est à disposition. Un système de dépose de fichier permet de télécharger le fichier csv contenant les données à prédire.
Les données téléchargées apparaissent sous format de tableau. Une fois la prédiction effectuée via le bouton de prédiction, une image indiquant la valeur de la prédiction apparaît pour chaque ligne (égal à un patient):
- soit un <img src="webapp/images/heart-svgrepo-com.svg" width="20" height="20" />  indiquant que la personne ne va pas mourir,
- ou une <img src="webapp/images/death-skull-and-bones-svgrepo-com.svg" width="20" height="20" />  indiquant sa mort prochaine.

## Étape 5 : Création d'un endpoint de feedback pour la mise à jour du modèle
Afin de permettre une correction du modèle en cas d'un nombre de mauvaises prédictions trop important, un endpoint de feedback permet à l'utilisateur de labelliser lui-même la donnée afin de corriger la prédiction du modèle. En validant pour une donnée sa prédiction, un nouveau fichier dit de production (prod_data.csv) est créé, contenant les lignes labellisées au fur et à mesure par l'utilisateur. Le fichier possède la même structure que le fichier de réference, à ceci près que dans la colonne dit de **target** ce sont les valeurs de feedback de l'utilisateur qui sont présentes, et qu'une colonne supplémentaire nommée **prediction** est ajoutée à côté et correspondant aux prédictions faîtes par le modèle.

Par ailleurs, un trigger lié au nombre de données nouvellement labélisées par l'utilisateur permet de déclencher le réentraînement du modèle et donc la mise à jour du fichier pickle. Si le nombre de lignes labélisées par l'utilisateur excède 20 (inclus), alors le réentraînement du modèle se lance en prenant en compte le fichier ref_data concaténé des données ajoutées de production.
Le fichier de production ainsi créé sera aussi utilisé pour la partie portant sur le reporting.

## Étape 6 : Système de reporting avec les dashboards et rapports Evidently
Afin d'obtenir des valeurs sur la performance de prédiction actuelle du modèle, un système de reporting (fixe car dépendant des fichiers de production générés en local lors d'un feedback de plusieurs lignes, à l'instar de Cronjob qui aurait permis un reporting régulier) est disponible grâce à l'utilisation d'Evidently. Via un dashboard et un rapport, l'interface permet l'affichage entre autres des Data Drifts, cad des changements importants dans la distribution des données entre le fichier de référence et celles nouvellement utilisée en production lors de leur prédiction. On retrouve notamment l'analyse de ces drifts pour 5 variables, qui sont les 5 premières issus du classement des variables les plus importantes dans notre dataset (ceci grâce à la lecture du fichier [features_sorted_by_importance.csv](data/features_sorted_by_importance.csv) stockant les variables classées afin de ne pas réexécuter la fonction de classement d'importance).

Des valeurs de métriques sont aussi disponibles :
- F1 score
- Accuracy
- Recall
- Precision
- Balanced Accuracy, non disponible de base parmi les métrqiues Evidently, pour laquelle il a été fait usage de la classe [*CustomValueMetric*](reporting/project.py#L65) permettant d'implémenter ses propres métriques en définissant une fonction de calcul de la métrique souhaitée.

Pour permettre cela, il est nécesssaire de noté qu'il faut que les fichiers de production et de réference ait la même structure ! Ce qui n'est pas le cas pour le fichier de référence créé seulement avec la colonne de target. Ainsi lors de la création du rapport Evidently, le fichier de référence chargé [se voit attribuer fictivement une colonne de prédiction](reporting/project.py#L33) correspondant à l'application de la méthode predict via le pipeline du modèle.

## Bonus : Système de monitoring en temps réel avec Prometheus et Grafana
Un système de [monitoring](/monitoring) a pu être mis en place en utilisant Prometheus et Grafana.
Prometheus permet la surveillance notamment des différents conteneurs Docker utilisés, mais pourrait aussi permettre la visualisation des métriques affichés dans le dashboard Evidently (non-réalisé). Quant à Grafana, il s'agit uniquement d'une interface de visualisation des données que Prometheus récupère, ici entre autre sur le runtime de chaque conteneur. À noter l'utilisation du conteneur `cadvisor` permettant de collecter de manière supplémentaire de données liées à l'utilisation du CPU, de la mémoire ou encore du réseau pour chaque conteneur du projet.

# 3) Vidéo de démo

[![Vidéo de démonstration du fonctionnement de l'API de serving](https://img.youtube.com/vi/Xnti5gW2xnI/0.jpg)](https://www.youtube.com/embed/Xnti5gW2xnI?si=2OESsGzo--7DTJpW)

# 4) Tests
Dans le projet vous trouverez un dossier [test_prediction](/test_prediction) contenant des fichiers csv que vous pouvez télécharger sur la webapp comme données à prédire.
Voici pour chaque fichier la target initiale (prédiction à obtenir sachant que 0 = <img src="webapp/images/heart-svgrepo-com.svg" width="20" height="20" /> et 1 = <img src="webapp/images/death-skull-and-bones-svgrepo-com.svg" width="20" height="20" /> ):

- pour [test_to_predict1.csv](/test_prediction/test_to_predict1.csv) (issu des 20 premières ligne de init_data)

| N° ligne | Target initiale |
|----------|-----------------|
| 1        | 0               |
| 2        | 0               |
| 3        | 0               |
| 4        | 0               |
| 5        | 0               |
| 6        | 0               |
| 7        | 0               |
| 8        | 0               |
| 9        | 1               |
| 10       | 0               |
| 11       | 0               |
| 12       | 0               |
| 13       | 0               |
| 14       | 0               |
| 15       | 0               |
| 16       | 0               |
| 17       | 0               |
| 18       | 0               |
| 19       | 0               |
| 20       | 0               |

- pour [test_to_predict2.csv](/test_prediction/test_to_predict2.csv) (issu des lignes 45090 à 45109 de init_data)

| N° ligne | Target initiale |
|----------|-----------------|
| 1        | 1               |
| 2        | 0               |
| 3        | 1               |
| 4        | 0               |
| 5        | 0               |
| 6        | 0               |
| 7        | 0               |
| 8        | 0               |
| 9        | 0               |
| 10       | 0               |
| 11       | 0               |
| 12       | 0               |
| 13       | 1               |
| 14       | 0               |
| 15       | 1               |
| 16       | 0               |
| 17       | 0               |
| 18       | 0               |
| 19       | 0               |
| 20       | 0               |

- pour [test_to_predict3.csv](/test_prediction/test_to_predict3.csv) (issu des lignes 45400 à 45429 de init_data)

| N° ligne | Target initiale |
|----------|-----------------|
| 1        | 0               |
| 2        | 0               |
| 3        | 0               |
| 4        | 0               |
| 5        | 0               |
| 6        | 0               |
| 7        | 0               |
| 8        | 0               |
| 9        | 0               |
| 10       | 0               |
| 11       | 1               |
| 12       | 0               |
| 13       | 0               |
| 14       | 0               |
| 15       | 0               |
| 16       | 0               |
| 17       | 0               |
| 18       | 0               |
| 19       | 0               |
| 20       | 0               |
| 21       | 1               |
| 22       | 0               |
| 23       | 0               |
| 24       | 0               |
| 25       | 0               |
| 26       | 0               |
| 27       | 0               |
| 28       | 0               |
| 29       | 0               |
| 30       | 0               |