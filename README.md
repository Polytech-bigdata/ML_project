# ML_project
Serving API for predictive machine learning model


## variable categoriel :

| Nom de la variable     | Description                                              | Valeurs possibles                                                                 |
|------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------|
| `ethnicity`            | Ethnicité ou appartenance culturelle du patient.         | Caucasian, African American, Other/Unknown, Hispanic, Asian                       |
| `gender`               | Sexe du patient.                                         | M, F                                                                              |
| `icu_admit_source`     | Source d'admission avant l'entrée en soins intensifs.    | Accident & Emergency, Operating Room / Recovery, Floor, Other Hospital, Other ICU |
| `icu_stay_type`        | Type de séjour en soins intensifs.                       | admit, transfer, readmit                                                          |
| `icu_type`             | Type d'unité de soins intensifs où le patient est admis. | Med-Surg ICU, MICU, Neuro ICU, CCU-CTICU, SICU                                    |
| `apache_3j_bodysystem` | Système corporel affecté selon APACHE III-J.             | Cardiovascular, Neurological, Sepsis, Respiratory, Gastrointestinal               |
| `apache_2_bodysystem`  | Système corporel affecté selon APACHE II.                | Cardiovascular, Neurologic, Respiratory, Gastrointestinal, Metabolic              |


| Nom de la variable              | Description                                              | Valeurs possibles                                                                 |
|----------------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------|
| `encounter_id`                   | Identifiant unique du séjour en unité.                   | -                                                                                 |
| `patient_id`                     | Identifiant unique du patient.                           | -                                                                                 |
| `hospital_id`                     | Identifiant unique de l'hôpital.                         | -                                                                                 |
| `age`                             | Âge du patient à l'admission en unité.                   | -                                                                                 |
| `bmi`                             | Indice de masse corporelle du patient.                   | -                                                                                 |
| `elective_surgery`                | Admission pour chirurgie programmée.                     | 0 (Non), 1 (Oui)                                                                 |
| `ethnicity`                       | Ethnicité ou appartenance culturelle du patient.         | Caucasian, African American, Other/Unknown, Hispanic, Asian                      |
| `gender`                          | Sexe du patient.                                         | M, F                                                                              |
| `height`                          | Taille du patient.                                       | -                                                                                 |
| `icu_admit_source`                | Source d'admission avant l'entrée en soins intensifs.    | Accident & Emergency, Operating Room / Recovery, Floor, Other Hospital, Other ICU |
| `icu_id`                          | Identifiant unique de l’unité de soins intensifs.        | -                                                                                 |
| `icu_stay_type`                   | Type de séjour en soins intensifs.                       | admit, transfer, readmit                                                          |
| `icu_type`                        | Type d'unité de soins intensifs où le patient est admis. | Med-Surg ICU, MICU, Neuro ICU, CCU-CTICU, SICU                                    |
| `pre_icu_los_days`                | Durée du séjour avant admission en soins intensifs.     | -                                                                                 |
| `weight`                          | Poids du patient.                                        | -                                                                                 |
| `apache_2_diagnosis`              | Code de diagnostic APACHE II.                           | -                                                                                 |
| `apache_3j_diagnosis`             | Code de diagnostic APACHE III-J.                        | -                                                                                 |
| `apache_post_operative`           | Statut post-opératoire APACHE.                          | 0 (Non), 1 (Oui)                                                                 |
| `arf_apache`                      | Présence d'insuffisance rénale aiguë.                    | 0 (Non), 1 (Oui)                                                                 |
| `gcs_eyes_apache`                 | Score APACHE de l’ouverture des yeux (GCS).             | 1-4                                                                              |
| `gcs_motor_apache`                | Score APACHE de la réponse motrice (GCS).               | 1-6                                                                              |
| `gcs_unable_apache`               | Évaluation GCS impossible (ex: sédation).               | 0 (Non), 1 (Oui)                                                                 |
| `gcs_verbal_apache`               | Score APACHE de la réponse verbale (GCS).               | 1-5                                                                              |
| `intubated_apache`                | Patient intubé lors des gaz du sang les plus sévères.   | 0 (Non), 1 (Oui)                                                                 |
| `ventilated_apache`               | Patient ventilé mécaniquement.                          | 0 (Non), 1 (Oui)                                                                 |
| `d1_diasbp_max`                   | Pression artérielle diastolique max, jour 1.            | -                                                                                 |
| `d1_diasbp_min`                   | Pression artérielle diastolique min, jour 1.            | -                                                                                 |
| `d1_diasbp_mean`                  | Pression artérielle diastolique moyenne, jour 1.        | -                                                                                 |
| `d1_heartrate_max`                | Fréquence cardiaque max, jour 1.                        | -                                                                                 |
| `d1_heartrate_min`                | Fréquence cardiaque min, jour 1.                        | -                                                                                 |
| `d1_heartrate_mean`               | Fréquence cardiaque moyenne, jour 1.                    | -                                                                                 |
| `d1_resprate_max`                 | Fréquence respiratoire max, jour 1.                     | -                                                                                 |
| `d1_resprate_min`                 | Fréquence respiratoire min, jour 1.                     | -                                                                                 |
| `d1_resprate_mean`                | Fréquence respiratoire moyenne, jour 1.                 | -                                                                                 |
| `d1_spo2_max`                     | Saturation en oxygène max, jour 1.                      | -                                                                                 |
| `d1_spo2_min`                     | Saturation en oxygène min, jour 1.                      | -                                                                                 |
| `d1_spo2_mean`                    | Saturation en oxygène moyenne, jour 1.                  | -                                                                                 |
| `d1_sysbp_max`                    | Pression artérielle systolique max, jour 1.             | -                                                                                 |
| `d1_sysbp_min`                    | Pression artérielle systolique min, jour 1.             | -                                                                                 |
| `d1_sysbp_mean`                   | Pression artérielle systolique moyenne, jour 1.         | -                                                                                 |
| `apache_4a_hospital_death_prob`   | Probabilité de décès à l’hôpital selon APACHE IVa.      | -                                                                                 |
| `apache_4a_icu_death_prob`        | Probabilité de décès en soins intensifs selon APACHE IVa. | -                                                                              |
| `aids`                            | Patient atteint du SIDA.                                | 0 (Non), 1 (Oui)                                                                 |
| `cirrhosis`                       | Présence d'une cirrhose.                                | 0 (Non), 1 (Oui)                                                                 |
| `diabetes_mellitus`               | Diabète nécessitant un traitement.                      | 0 (Non), 1 (Oui)                                                                 |
| `hepatic_failure`                 | Insuffisance hépatique sévère.                          | 0 (Non), 1 (Oui)                                                                 |
| `immunosuppression`               | Patient immunodéprimé avant admission.                  | 0 (Non), 1 (Oui)                                                                 |
| `leukemia`                        | Présence d'une leucémie.                                | 0 (Non), 1 (Oui)                                                                 |
| `lymphoma`                        | Présence d'un lymphome.                                | 0 (Non), 1 (Oui)                                                                 |
| `solid_tumor_with_metastasis`     | Cancer avec métastases.                                | 0 (Non), 1 (Oui)                                                                 |
| `apache_3j_bodysystem`            | Système corporel affecté selon APACHE III-J.           | Cardiovascular, Neurological, Sepsis, Respiratory, Gastrointestinal               |
| `apache_2_bodysystem`             | Système corporel affecté selon APACHE II.              | Cardiovascular, Neurologic, Respiratory, Gastrointestinal, Metabolic              |
| `hospital_death`                  | Statut du patient à la sortie de l'hôpital.            | 0 (Survivant), 1 (Décédé)                                                         |
