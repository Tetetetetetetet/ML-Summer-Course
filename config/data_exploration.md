50 features: ['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'payer_code', 'medical_specialty', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted', 'diag_1', 'diag_2', 'diag_3']

## race: (demographic, categorical)

race 与 readmitted 的皮尔逊相关系数为: -0.014938392550048243

## gender: (demographic, categorical)

gender 与 readmitted 的皮尔逊相关系数为: 0.013667127725071388

## age: (demographic, categorical)

age 与 readmitted 的皮尔逊相关系数为: -0.03863727565178263

## weight: (demographic, categorical)

weight 与 readmitted 的皮尔逊相关系数为: -0.03170967649167792

## admission_type_id: (categorical, categorical)

admission_type_id
 0.0    47625
 2.0    16794
 1.0    16459
-1.0     9201
 4.0       16
 3.0       10
Name: count, dtype: int64

{0: '1', 1: '2', 2: '3', 3: '4', 4: '7'}

### statics

|    |   class | class_name   |    mean |      std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------|--------:|---------:|------:|------:|---------:|-------:|--------:|
|  0 |      -1 | N/A          | 1.38441 | 0.67416  |     0 |     2 |        1 |      2 |    9201 |
|  1 |       0 | 1            | 1.3999  | 0.689196 |     0 |     2 |        2 |      2 |   47625 |
|  2 |       1 | 2            | 1.41691 | 0.687153 |     0 |     2 |        2 |      2 |   16459 |
|  3 |       2 | 3            | 1.48589 | 0.677898 |     0 |     2 |        2 |      2 |   16794 |
|  4 |       3 | 4            | 1.6     | 0.699206 |     0 |     2 |        2 |      2 |      10 |
|  5 |       4 | 7            | 2       | 0        |     2 |     2 |        2 |      2 |      16 |

admission_type_id 与 readmitted 的相关系数矩阵：

|                   |   admission_type_id |   readmitted |
|:------------------|--------------------:|-------------:|
| admission_type_id |             1       |      0.04679 |
| readmitted        |             0.04679 |      1       |

admission_type_id 与 readmitted 的皮尔逊相关系数为: 0.04678995785809007

## discharge_disposition_id: (categorical, categorical)

discharge_disposition_id
 0.0     54187
 14.0    12607
 17.0    11589
-1.0      4223
 8.0      1893
 9.0      1791
 16.0     1071
 15.0      738
 18.0      557
 10.0      380
 3.0       346
 4.0       341
 13.0      130
 19.0       98
 5.0        59
 11.0       42
 20.0       19
 7.0        12
 6.0         9
 1.0         6
 12.0        4
 2.0         3
Name: count, dtype: int64

{0: '1', 1: '10', 2: '12', 3: '13', 4: '14', 5: '15', 6: '16', 7: '17', 8: '2', 9: '22', 10: '23', 11: '24', 12: '27', 13: '28', 14: '3', 15: '4', 16: '5', 17: '6', 18: '7', 19: '8', 20: '9'}

### statics

|    |   class | class_name   |     mean |      std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------|---------:|---------:|------:|------:|---------:|-------:|--------:|
|  0 |      -1 | N/A          | 1.46081  | 0.69669  |     0 |     2 |        2 |      2 |    4223 |
|  1 |       0 | 1            | 1.45743  | 0.658953 |     0 |     2 |        2 |      2 |   54187 |
|  2 |       1 | 10           | 1.33333  | 0.516398 |     1 |     2 |        1 |      1 |       6 |
|  3 |       2 | 12           | 0.666667 | 1.1547   |     0 |     2 |        0 |      0 |       3 |
|  4 |       3 | 13           | 1.80925  | 0.509078 |     0 |     2 |        2 |      2 |     346 |
|  5 |       4 | 14           | 1.84457  | 0.516671 |     0 |     2 |        2 |      2 |     341 |
|  6 |       5 | 15           | 0.864407 | 0.839902 |     0 |     2 |        1 |      0 |      59 |
|  7 |       6 | 16           | 1.44444  | 0.527046 |     1 |     2 |        1 |      1 |       9 |
|  8 |       7 | 17           | 1.66667  | 0.492366 |     1 |     2 |        2 |      2 |      12 |
|  9 |       8 | 2            | 1.35816  | 0.743956 |     0 |     2 |        2 |      2 |    1893 |
| 10 |       9 | 22           | 1.18816  | 0.842843 |     0 |     2 |        1 |      2 |    1791 |
| 11 |      10 | 23           | 1.5      | 0.627009 |     0 |     2 |        2 |      2 |     380 |
| 12 |      11 | 24           | 1.40476  | 0.700506 |     0 |     2 |        2 |      2 |      42 |
| 13 |      12 | 27           | 1.75     | 0.5      |     1 |     2 |        2 |      2 |       4 |
| 14 |      13 | 28           | 1.01538  | 0.880315 |     0 |     2 |        1 |      2 |     130 |
| 15 |      14 | 3            | 1.35504  | 0.722084 |     0 |     2 |        2 |      2 |   12607 |
| 16 |      15 | 4            | 1.42141  | 0.696903 |     0 |     2 |        2 |      2 |     738 |
| 17 |      16 | 5            | 1.27638  | 0.791421 |     0 |     2 |        1 |      2 |    1071 |
| 18 |      17 | 6            | 1.32522  | 0.691012 |     0 |     2 |        1 |      2 |   11589 |
| 19 |      18 | 7            | 1.35727  | 0.712053 |     0 |     2 |        1 |      2 |     557 |
| 20 |      19 | 8            | 1.41837  | 0.687495 |     0 |     2 |        2 |      2 |      98 |
| 21 |      20 | 9            | 1.05263  | 0.97032  |     0 |     2 |        1 |      2 |      19 |

discharge_disposition_id 与 readmitted 的相关系数矩阵：

|                          |   discharge_disposition_id |   readmitted |
|:-------------------------|---------------------------:|-------------:|
| discharge_disposition_id |                  1         |   -0.0831123 |
| readmitted               |                 -0.0831123 |    1         |

discharge_disposition_id 与 readmitted 的皮尔逊相关系数为: -0.08311230005301229

## admission_source_id: (categorical, categorical)

admission_source_id
 12.0    50824
 0.0     26347
-1.0      6233
 9.0      2786
 11.0     2019
 5.0       963
 10.0      722
 8.0       171
 13.0       14
 6.0        12
 1.0         8
 4.0         2
 7.0         2
 2.0         1
 3.0         1
Name: count, dtype: int64

{0: '1', 1: '10', 2: '11', 3: '13', 4: '14', 5: '2', 6: '22', 7: '25', 8: '3', 9: '4', 10: '5', 11: '6', 12: '7', 13: '8'}

### statics

|    |   class | class_name   |    mean |        std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------|--------:|-----------:|------:|------:|---------:|-------:|--------:|
|  0 |      -1 | N/A          | 1.41264 |   0.676096 |     0 |     2 |        2 |      2 |    6233 |
|  1 |       0 | 1            | 1.45834 |   0.679906 |     0 |     2 |        2 |      2 |   26347 |
|  2 |       1 | 10           | 1.75    |   0.46291  |     1 |     2 |        2 |      2 |       8 |
|  3 |       2 | 11           | 2       | nan        |     2 |     2 |        2 |      2 |       1 |
|  4 |       3 | 13           | 2       | nan        |     2 |     2 |        2 |      2 |       1 |
|  5 |       4 | 14           | 2       |   0        |     2 |     2 |        2 |      2 |       2 |
|  6 |       5 | 2            | 1.5026  |   0.672247 |     0 |     2 |        2 |      2 |     963 |
|  7 |       6 | 22           | 1.41667 |   0.792961 |     0 |     2 |        2 |      2 |      12 |
|  8 |       7 | 25           | 2       |   0        |     2 |     2 |        2 |      2 |       2 |
|  9 |       8 | 3            | 1.39181 |   0.746555 |     0 |     2 |        2 |      2 |     171 |
| 10 |       9 | 4            | 1.58543 |   0.664614 |     0 |     2 |        2 |      2 |    2786 |
| 11 |      10 | 5            | 1.47784 |   0.694885 |     0 |     2 |        2 |      2 |     722 |
| 12 |      11 | 6            | 1.63744 |   0.646945 |     0 |     2 |        2 |      2 |    2019 |
| 13 |      12 | 7            | 1.37657 |   0.6885   |     0 |     2 |        1 |      2 |   50824 |
| 14 |      13 | 8            | 1.57143 |   0.646206 |     0 |     2 |        2 |      2 |      14 |

admission_source_id 与 readmitted 的相关系数矩阵：

|                     |   admission_source_id |   readmitted |
|:--------------------|----------------------:|-------------:|
| admission_source_id |             1         |   -0.0421241 |
| readmitted          |            -0.0421241 |    1         |

admission_source_id 与 readmitted 的皮尔逊相关系数为: -0.042124139176484734

## time_in_hospital: (numeric, continuous)

time_in_hospital 与 readmitted 的皮尔逊相关系数为: -0.06198172387210785

## payer_code: (categorical, categorical)

payer_code
-1.0     35646
 7.0     28529
 6.0      5610
 14.0     4474
 0.0      4140
 8.0      3160
 3.0      2252
 15.0     2189
 2.0      1710
 10.0      921
 12.0      525
 4.0       487
 1.0       130
 16.0      126
 11.0       82
 9.0        70
 13.0       53
 5.0         1
Name: count, dtype: int64

{0: 'BC', 1: 'CH', 2: 'CM', 3: 'CP', 4: 'DM', 5: 'FR', 6: 'HM', 7: 'MC', 8: 'MD', 9: 'MP', 10: 'OG', 11: 'OT', 12: 'PO', 13: 'SI', 14: 'SP', 15: 'UN', 16: 'WC'}

### statics

|    |   class | class_name   |    mean |        std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------|--------:|-----------:|------:|------:|---------:|-------:|--------:|
|  0 |      -1 | N/A          | 1.41483 |   0.691185 |     0 |     2 |        2 |      2 |   35646 |
|  1 |       0 | BC           | 1.53213 |   0.659227 |     0 |     2 |        2 |      2 |    4140 |
|  2 |       1 | CH           | 1.57692 |   0.657283 |     0 |     2 |        2 |      2 |     130 |
|  3 |       2 | CM           | 1.4345  |   0.679106 |     0 |     2 |        2 |      2 |    1710 |
|  4 |       3 | CP           | 1.52087 |   0.650332 |     0 |     2 |        2 |      2 |    2252 |
|  5 |       4 | DM           | 1.34908 |   0.688914 |     0 |     2 |        1 |      2 |     487 |
|  6 |       5 | FR           | 2       | nan        |     2 |     2 |        2 |      2 |       1 |
|  7 |       6 | HM           | 1.42157 |   0.669749 |     0 |     2 |        2 |      2 |    5610 |
|  8 |       7 | MC           | 1.3828  |   0.690717 |     0 |     2 |        2 |      2 |   28529 |
|  9 |       8 | MD           | 1.40475 |   0.689387 |     0 |     2 |        2 |      2 |    3160 |
| 10 |       9 | MP           | 1.28571 |   0.662513 |     0 |     2 |        1 |      1 |      70 |
| 11 |      10 | OG           | 1.40717 |   0.711942 |     0 |     2 |        2 |      2 |     921 |
| 12 |      11 | OT           | 1.45122 |   0.611665 |     0 |     2 |        2 |      2 |      82 |
| 13 |      12 | PO           | 1.60381 |   0.623326 |     0 |     2 |        2 |      2 |     525 |
| 14 |      13 | SI           | 1.43396 |   0.720828 |     0 |     2 |        2 |      2 |      53 |
| 15 |      14 | SP           | 1.42155 |   0.670839 |     0 |     2 |        2 |      2 |    4474 |
| 16 |      15 | UN           | 1.52764 |   0.655304 |     0 |     2 |        2 |      2 |    2189 |
| 17 |      16 | WC           | 1.7619  |   0.496847 |     0 |     2 |        2 |      2 |     126 |

payer_code 与 readmitted 的相关系数矩阵：

|            |   payer_code |   readmitted |
|:-----------|-------------:|-------------:|
| payer_code |   1          |  -0.00267444 |
| readmitted |  -0.00267444 |   1          |

payer_code 与 readmitted 的皮尔逊相关系数为: -0.0026744410724981282

## medical_specialty: (categorical, categorical)

medical_specialty
-1.0     44259
 18.0    12913
 8.0      6698
 11.0     6590
 3.0      4729
         ...  
 48.0        1
 42.0        1
 21.0        1
 39.0        1
 67.0        1
Name: count, Length: 73, dtype: int64

{0: 'AllergyandImmunology', 1: 'Anesthesiology', 2: 'Anesthesiology-Pediatric', 3: 'Cardiology', 4: 'Cardiology-Pediatric', 5: 'DCPTEAM', 6: 'Dentistry', 7: 'Dermatology', 8: 'Emergency/Trauma', 9: 'Endocrinology', 10: 'Endocrinology-Metabolism', 11: 'Family/GeneralPractice', 12: 'Gastroenterology', 13: 'Gynecology', 14: 'Hematology', 15: 'Hematology/Oncology', 16: 'Hospitalist', 17: 'InfectiousDiseases', 18: 'InternalMedicine', 19: 'Nephrology', 20: 'Neurology', 21: 'Neurophysiology', 22: 'Obsterics&Gynecology-GynecologicOnco', 23: 'Obstetrics', 24: 'ObstetricsandGynecology', 25: 'Oncology', 26: 'Ophthalmology', 27: 'Orthopedics', 28: 'Orthopedics-Reconstructive', 29: 'Osteopath', 30: 'Otolaryngology', 31: 'OutreachServices', 32: 'Pathology', 33: 'Pediatrics', 34: 'Pediatrics-AllergyandImmunology', 35: 'Pediatrics-CriticalCare', 36: 'Pediatrics-EmergencyMedicine', 37: 'Pediatrics-Endocrinology', 38: 'Pediatrics-Hematology-Oncology', 39: 'Pediatrics-InfectiousDiseases', 40: 'Pediatrics-Neurology', 41: 'Pediatrics-Pulmonology', 42: 'Perinatology', 43: 'PhysicalMedicineandRehabilitation', 44: 'PhysicianNotFound', 45: 'Podiatry', 46: 'Proctology', 47: 'Psychiatry', 48: 'Psychiatry-Addictive', 49: 'Psychiatry-Child/Adolescent', 50: 'Psychology', 51: 'Pulmonology', 52: 'Radiologist', 53: 'Radiology', 54: 'Resident', 55: 'Rheumatology', 56: 'Speech', 57: 'SportsMedicine', 58: 'Surgeon', 59: 'Surgery-Cardiovascular', 60: 'Surgery-Cardiovascular/Thoracic', 61: 'Surgery-Colon&Rectal', 62: 'Surgery-General', 63: 'Surgery-Maxillofacial', 64: 'Surgery-Neuro', 65: 'Surgery-Pediatric', 66: 'Surgery-Plastic', 67: 'Surgery-PlasticwithinHeadandNeck', 68: 'Surgery-Thoracic', 69: 'Surgery-Vascular', 70: 'SurgicalSpecialty', 71: 'Urology'}

### statics

|    |   class | class_name                           |     mean |        std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------------------------------|---------:|-----------:|------:|------:|---------:|-------:|--------:|
|  0 |      -1 | N/A                                  | 1.39341  |   0.6887   |     0 |     2 |      2   |      2 |   44259 |
|  1 |       0 | AllergyandImmunology                 | 0.714286 |   0.755929 |     0 |     2 |      1   |      0 |       7 |
|  2 |       1 | Anesthesiology                       | 1.63636  |   0.6742   |     0 |     2 |      2   |      2 |      11 |
|  3 |       2 | Anesthesiology-Pediatric             | 1.61111  |   0.607685 |     0 |     2 |      2   |      2 |      18 |
|  4 |       3 | Cardiology                           | 1.48678  |   0.641458 |     0 |     2 |      2   |      2 |    4729 |
|  5 |       4 | Cardiology-Pediatric                 | 1.14286  |   0.690066 |     0 |     2 |      1   |      1 |       7 |
|  6 |       5 | DCPTEAM                              | 1.8      |   0.447214 |     1 |     2 |      2   |      2 |       5 |
|  7 |       6 | Dentistry                            | 1.25     |   0.5      |     1 |     2 |      1   |      1 |       4 |
|  8 |       7 | Dermatology                          | 1        | nan        |     1 |     1 |      1   |      1 |       1 |
|  9 |       8 | Emergency/Trauma                     | 1.36698  |   0.678382 |     0 |     2 |      1   |      2 |    6698 |
| 10 |       9 | Endocrinology                        | 1.55238  |   0.604167 |     0 |     2 |      2   |      2 |     105 |
| 11 |      10 | Endocrinology-Metabolism             | 1.8      |   0.447214 |     1 |     2 |      2   |      2 |       5 |
| 12 |      11 | Family/GeneralPractice               | 1.39287  |   0.692069 |     0 |     2 |      2   |      2 |    6590 |
| 13 |      12 | Gastroenterology                     | 1.3963   |   0.667458 |     0 |     2 |      1   |      2 |     487 |
| 14 |      13 | Gynecology                           | 1.83333  |   0.376177 |     1 |     2 |      2   |      2 |      54 |
| 15 |      14 | Hematology                           | 1.02817  |   0.844678 |     0 |     2 |      1   |      2 |      71 |
| 16 |      15 | Hematology/Oncology                  | 1.27072  |   0.780793 |     0 |     2 |      1   |      2 |     181 |
| 17 |      16 | Hospitalist                          | 1.54902  |   0.610368 |     0 |     2 |      2   |      2 |      51 |
| 18 |      17 | InfectiousDiseases                   | 1.17647  |   0.796606 |     0 |     2 |      1   |      2 |      34 |
| 19 |      18 | InternalMedicine                     | 1.4425   |   0.690345 |     0 |     2 |      2   |      2 |   12913 |
| 20 |      19 | Nephrology                           | 1.23444  |   0.71502  |     0 |     2 |      1   |      1 |    1382 |
| 21 |      20 | Neurology                            | 1.67403  |   0.56651  |     0 |     2 |      2   |      2 |     181 |
| 22 |      21 | Neurophysiology                      | 2        | nan        |     2 |     2 |      2   |      2 |       1 |
| 23 |      22 | Obsterics&Gynecology-GynecologicOnco | 1.6      |   0.680557 |     0 |     2 |      2   |      2 |      20 |
| 24 |      23 | Obstetrics                           | 1.77778  |   0.548319 |     0 |     2 |      2   |      2 |      18 |
| 25 |      24 | ObstetricsandGynecology              | 1.73422  |   0.546468 |     0 |     2 |      2   |      2 |     602 |
| 26 |      25 | Oncology                             | 1.26761  |   0.792326 |     0 |     2 |      1   |      2 |     284 |
| 27 |      26 | Ophthalmology                        | 1.5625   |   0.618922 |     0 |     2 |      2   |      2 |      32 |
| 28 |      27 | Orthopedics                          | 1.55425  |   0.67574  |     0 |     2 |      2   |      2 |    1281 |
| 29 |      28 | Orthopedics-Reconstructive           | 1.60795  |   0.629195 |     0 |     2 |      2   |      2 |    1107 |
| 30 |      29 | Osteopath                            | 1.39394  |   0.609272 |     0 |     2 |      1   |      1 |      33 |
| 31 |      30 | Otolaryngology                       | 1.75     |   0.454447 |     0 |     2 |      2   |      2 |     116 |
| 32 |      31 | OutreachServices                     | 1.41667  |   0.668558 |     0 |     2 |      1.5 |      2 |      12 |
| 33 |      32 | Pathology                            | 1.21429  |   0.578934 |     0 |     2 |      1   |      1 |      14 |
| 34 |      33 | Pediatrics                           | 1.57269  |   0.615258 |     0 |     2 |      2   |      2 |     227 |
| 35 |      34 | Pediatrics-AllergyandImmunology      | 1        |   0        |     1 |     1 |      1   |      1 |       2 |
| 36 |      35 | Pediatrics-CriticalCare              | 1.62025  |   0.561678 |     0 |     2 |      2   |      2 |      79 |
| 37 |      36 | Pediatrics-EmergencyMedicine         | 1.66667  |   0.57735  |     1 |     2 |      2   |      2 |       3 |
| 38 |      37 | Pediatrics-Endocrinology             | 1.84507  |   0.382151 |     0 |     2 |      2   |      2 |     142 |
| 39 |      38 | Pediatrics-Hematology-Oncology       | 1.5      |   1        |     0 |     2 |      2   |      2 |       4 |
| 40 |      39 | Pediatrics-InfectiousDiseases        | 1        | nan        |     1 |     1 |      1   |      1 |       1 |
| 41 |      40 | Pediatrics-Neurology                 | 1.6      |   0.516398 |     1 |     2 |      2   |      2 |      10 |
| 42 |      41 | Pediatrics-Pulmonology               | 1.19048  |   0.601585 |     0 |     2 |      1   |      1 |      21 |
| 43 |      42 | Perinatology                         | 2        | nan        |     2 |     2 |      2   |      2 |       1 |
| 44 |      43 | PhysicalMedicineandRehabilitation    | 1.45665  |   0.753446 |     0 |     2 |      2   |      2 |     346 |
| 45 |      44 | PhysicianNotFound                    | 1.2      |   0.788811 |     0 |     2 |      1   |      1 |      10 |
| 46 |      45 | Podiatry                             | 1.30588  |   0.673009 |     0 |     2 |      1   |      1 |      85 |
| 47 |      46 | Proctology                           | 2        | nan        |     2 |     2 |      2   |      2 |       1 |
| 48 |      47 | Psychiatry                           | 1.44386  |   0.705799 |     0 |     2 |      2   |      2 |     766 |
| 49 |      48 | Psychiatry-Addictive                 | 2        | nan        |     2 |     2 |      2   |      2 |       1 |
| 50 |      49 | Psychiatry-Child/Adolescent          | 1.42857  |   0.786796 |     0 |     2 |      2   |      2 |       7 |
| 51 |      50 | Psychology                           | 1.50538  |   0.618969 |     0 |     2 |      2   |      2 |      93 |
| 52 |      51 | Pulmonology                          | 1.38947  |   0.681697 |     0 |     2 |      2   |      2 |     760 |
| 53 |      52 | Radiologist                          | 1.48718  |   0.652532 |     0 |     2 |      2   |      2 |    1014 |
| 54 |      53 | Radiology                            | 1.38298  |   0.767639 |     0 |     2 |      2   |      2 |      47 |
| 55 |      54 | Resident                             | 0.5      |   0.707107 |     0 |     1 |      0.5 |      0 |       2 |
| 56 |      55 | Rheumatology                         | 1.4375   |   0.727438 |     0 |     2 |      2   |      2 |      16 |
| 57 |      56 | Speech                               | 2        | nan        |     2 |     2 |      2   |      2 |       1 |
| 58 |      57 | SportsMedicine                       | 1        | nan        |     1 |     1 |      1   |      1 |       1 |
| 59 |      58 | Surgeon                              | 1.625    |   0.667467 |     0 |     2 |      2   |      2 |      40 |
| 60 |      59 | Surgery-Cardiovascular               | 1.62353  |   0.616759 |     0 |     2 |      2   |      2 |      85 |
| 61 |      60 | Surgery-Cardiovascular/Thoracic      | 1.67372  |   0.592411 |     0 |     2 |      2   |      2 |     567 |
| 62 |      61 | Surgery-Colon&Rectal                 | 1.6      |   0.699206 |     0 |     2 |      2   |      2 |      10 |
| 63 |      62 | Surgery-General                      | 1.43033  |   0.688939 |     0 |     2 |      2   |      2 |    2756 |
| 64 |      63 | Surgery-Maxillofacial                | 1.6      |   0.699206 |     0 |     2 |      2   |      2 |      10 |
| 65 |      64 | Surgery-Neuro                        | 1.70379  |   0.580715 |     0 |     2 |      2   |      2 |     422 |
| 66 |      65 | Surgery-Pediatric                    | 1.75     |   0.46291  |     1 |     2 |      2   |      2 |       8 |
| 67 |      66 | Surgery-Plastic                      | 1.4      |   0.7779   |     0 |     2 |      2   |      2 |      40 |
| 68 |      67 | Surgery-PlasticwithinHeadandNeck     | 1        | nan        |     1 |     1 |      1   |      1 |       1 |
| 69 |      68 | Surgery-Thoracic                     | 1.50515  |   0.694078 |     0 |     2 |      2   |      2 |      97 |
| 70 |      69 | Surgery-Vascular                     | 1.34454  |   0.710072 |     0 |     2 |      1   |      2 |     476 |
| 71 |      70 | SurgicalSpecialty                    | 1.64286  |   0.621485 |     0 |     2 |      2   |      2 |      28 |
| 72 |      71 | Urology                              | 1.53758  |   0.662455 |     0 |     2 |      2   |      2 |     612 |

medical_specialty 与 readmitted 的相关系数矩阵：

|                   |   medical_specialty |   readmitted |
|:------------------|--------------------:|-------------:|
| medical_specialty |           1         |    0.0429492 |
| readmitted        |           0.0429492 |    1         |

medical_specialty 与 readmitted 的皮尔逊相关系数为: 0.04294924272211373

## num_lab_procedures: (numeric, continuous)

num_lab_procedures 与 readmitted 的皮尔逊相关系数为: -0.04463745818879579

## num_procedures: (numeric, continuous)

num_procedures 与 readmitted 的皮尔逊相关系数为: 0.033733856067160166

## num_medications: (numeric, continuous)

num_medications 与 readmitted 的皮尔逊相关系数为: -0.05759238846594891

## number_outpatient: (numeric, continuous)

number_outpatient 与 readmitted 的皮尔逊相关系数为: -0.0701460709807518

## number_emergency: (numeric, continuous)

number_emergency 与 readmitted 的皮尔逊相关系数为: -0.10302319747285739

## number_inpatient: (numeric, continuous)

number_inpatient 与 readmitted 的皮尔逊相关系数为: -0.2402597244307448

## number_diagnoses: (numeric, continuous)

number_diagnoses 与 readmitted 的皮尔逊相关系数为: -0.11301054507903267

## max_glu_serum: (categorical, categorical)

max_glu_serum
-1.0    85420
 2.0     2280
 0.0     1305
 1.0     1100
Name: count, dtype: int64

{0: '>200', 1: '>300', 2: 'Norm'}

### statics

|    |   class | class_name   |    mean |      std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------|--------:|---------:|------:|------:|---------:|-------:|--------:|
|  0 |      -1 | N/A          | 1.42016 | 0.685271 |     0 |     2 |        2 |      2 |   85420 |
|  1 |       0 | >200         | 1.36552 | 0.702017 |     0 |     2 |        1 |      2 |    1305 |
|  2 |       1 | >300         | 1.27182 | 0.705965 |     0 |     2 |        1 |      1 |    1100 |
|  3 |       2 | Norm         | 1.42105 | 0.686416 |     0 |     2 |        2 |      2 |    2280 |

max_glu_serum 与 readmitted 的相关系数矩阵：

|               |   max_glu_serum |   readmitted |
|:--------------|----------------:|-------------:|
| max_glu_serum |       1         |   -0.0111977 |
| readmitted    |      -0.0111977 |    1         |

max_glu_serum 与 readmitted 的皮尔逊相关系数为: -0.011197684576482295

## A1Cresult: (categorical, categorical)

A1Cresult
-1.0    74928
 1.0     7339
 2.0     4434
 0.0     3404
Name: count, dtype: int64

{0: '>7', 1: '>8', 2: 'Norm'}

### statics

|    |   class | class_name   |    mean |      std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------|--------:|---------:|------:|------:|---------:|-------:|--------:|
|  0 |      -1 | N/A          | 1.40974 | 0.689001 |     0 |     2 |        2 |      2 |   74928 |
|  1 |       0 | >7           | 1.45065 | 0.67068  |     0 |     2 |        2 |      2 |    3404 |
|  2 |       1 | >8           | 1.44734 | 0.66902  |     0 |     2 |        2 |      2 |    7339 |
|  3 |       2 | Norm         | 1.47542 | 0.669974 |     0 |     2 |        2 |      2 |    4434 |

A1Cresult 与 readmitted 的相关系数矩阵：

|            |   A1Cresult |   readmitted |
|:-----------|------------:|-------------:|
| A1Cresult  |   1         |    0.0257734 |
| readmitted |   0.0257734 |    1         |

A1Cresult 与 readmitted 的皮尔逊相关系数为: 0.02577341204052467

## metformin: (medication, categorical)

metformin 与 readmitted 的皮尔逊相关系数为: 0.0373739809478137

## repaglinide: (medication, categorical)

repaglinide 与 readmitted 的皮尔逊相关系数为: -0.020194815588247213

## nateglinide: (medication, categorical)

nateglinide 与 readmitted 的皮尔逊相关系数为: -0.00393706172777648

## chlorpropamide: (medication, categorical)

chlorpropamide 与 readmitted 的皮尔逊相关系数为: 0.0009616596064394665

## glimepiride: (medication, categorical)

glimepiride 与 readmitted 的皮尔逊相关系数为: 0.004174491526749291

## acetohexamide: (medication, categorical)

acetohexamide 与 readmitted 的皮尔逊相关系数为: -0.0020278571954990203

## glipizide: (medication, categorical)

glipizide 与 readmitted 的皮尔逊相关系数为: -0.008856788663620462

## glyburide: (medication, categorical)

glyburide 与 readmitted 的皮尔逊相关系数为: 0.007218222263614739

## tolbutamide: (medication, categorical)

tolbutamide 与 readmitted 的皮尔逊相关系数为: 0.0028430152454772344

## pioglitazone: (medication, categorical)

pioglitazone 与 readmitted 的皮尔逊相关系数为: -0.0011438250942928214

## rosiglitazone: (medication, categorical)

rosiglitazone 与 readmitted 的皮尔逊相关系数为: -0.005905787166633014

## acarbose: (medication, categorical)

acarbose 与 readmitted 的皮尔逊相关系数为: -0.008036152160830024

## miglitol: (medication, categorical)

miglitol 与 readmitted 的皮尔逊相关系数为: 0.001558429169574278

## troglitazone: (medication, categorical)

troglitazone 与 readmitted 的皮尔逊相关系数为: -0.0007086205712180946

## tolazamide: (medication, categorical)

tolazamide 与 readmitted 的皮尔逊相关系数为: 0.006062894979230304

## examide: (medication, categorical)

examide 与 readmitted 的皮尔逊相关系数为: nan

## citoglipton: (medication, categorical)

citoglipton 与 readmitted 的皮尔逊相关系数为: nan

## insulin: (medication, categorical)

insulin 与 readmitted 的皮尔逊相关系数为: -0.0056511684744882705

## glyburide-metformin: (medication, categorical)

glyburide-metformin 与 readmitted 的皮尔逊相关系数为: 0.0012999674698094266

## glipizide-metformin: (medication, categorical)

glipizide-metformin 与 readmitted 的皮尔逊相关系数为: -0.0019241806608863085

## glimepiride-pioglitazone: (medication, categorical)

glimepiride-pioglitazone 与 readmitted 的皮尔逊相关系数为: nan

## metformin-rosiglitazone: (medication, categorical)

metformin-rosiglitazone 与 readmitted 的皮尔逊相关系数为: 0.0028283611801040104

## metformin-pioglitazone: (medication, categorical)

metformin-pioglitazone 与 readmitted 的皮尔逊相关系数为: 0.002828361180104045

## change: (categorical, categorical)

change
1    48256
0    41849
Name: count, dtype: int64

{0: 'Ch', 1: 'No'}

### statics

|    |   class | class_name   |    mean |      std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------|--------:|---------:|------:|------:|---------:|-------:|--------:|
|  0 |       0 | Ch           | 1.38854 | 0.691082 |     0 |     2 |        2 |      2 |   41849 |
|  1 |       1 | No           | 1.44276 | 0.680591 |     0 |     2 |        2 |      2 |   48256 |

change 与 readmitted 的相关系数矩阵：

|            |    change |   readmitted |
|:-----------|----------:|-------------:|
| change     | 1         |    0.0394212 |
| readmitted | 0.0394212 |    1         |

change 与 readmitted 的皮尔逊相关系数为: 0.03942120158842363

## diabetesMed: (categorical, categorical)

diabetesMed
1    69574
0    20531
Name: count, dtype: int64

{0: 'No', 1: 'Yes'}

### statics

|    |   class | class_name   |    mean |      std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------|--------:|---------:|------:|------:|---------:|-------:|--------:|
|  0 |       0 | No           | 1.48697 | 0.66801  |     0 |     2 |        2 |      2 |   20531 |
|  1 |       1 | Yes          | 1.3971  | 0.689908 |     0 |     2 |        2 |      2 |   69574 |

diabetesMed 与 readmitted 的相关系数矩阵：

|             |   diabetesMed |   readmitted |
|:------------|--------------:|-------------:|
| diabetesMed |     1         |   -0.0549487 |
| readmitted  |    -0.0549487 |    1         |

diabetesMed 与 readmitted 的皮尔逊相关系数为: -0.05494872945031728

## readmitted: (target, categorical)

readmitted 与 readmitted 的皮尔逊相关系数为:             readmitted  readmitted
readmitted         1.0         1.0
readmitted         1.0         1.0

## diag_1: (diagnosis, categorical)

diag_1 与 readmitted 的皮尔逊相关系数为: 0.031555166611784244

## diag_2: (diagnosis, categorical)

diag_2 与 readmitted 的皮尔逊相关系数为: 0.0028811799644389373

## diag_3: (diagnosis, categorical)

diag_3 与 readmitted 的皮尔逊相关系数为: -0.0170242309636612

