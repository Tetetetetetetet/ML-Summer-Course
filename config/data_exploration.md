50 features: ['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'payer_code', 'medical_specialty', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted', 'diag_1', 'diag_2', 'diag_3']

## race: (demographic, categorical)

race 与 readmitted 的皮尔逊相关系数为: 0.0174269532457985

## gender: (demographic, categorical)

gender 与 readmitted 的皮尔逊相关系数为: -0.01651084301036242

## age: (demographic, categorical)

age 与 readmitted 的皮尔逊相关系数为: 0.03522317189076101

## weight: (demographic, categorical)

weight 与 readmitted 的皮尔逊相关系数为: 0.046722554398117226

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

|    |   class | class_name   |     mean |      std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------|---------:|---------:|------:|------:|---------:|-------:|--------:|
|  0 |      -1 | N/A          | 0.904467 | 0.939185 |     0 |     2 |        1 |      0 |    9201 |
|  1 |       0 | 1            | 0.847706 | 0.926997 |     0 |     2 |        0 |      0 |   47625 |
|  2 |       1 | 2            | 0.822589 | 0.924148 |     0 |     2 |        0 |      0 |   16459 |
|  3 |       2 | 3            | 0.713648 | 0.901772 |     0 |     2 |        0 |      0 |   16794 |
|  4 |       3 | 4            | 0.5      | 0.849837 |     0 |     2 |        0 |      0 |      10 |
|  5 |       4 | 7            | 0        | 0        |     0 |     0 |        0 |      0 |      16 |

admission_type_id 与 readmitted 的相关系数矩阵：

|                   |   admission_type_id |   readmitted |
|:------------------|--------------------:|-------------:|
| admission_type_id |           1         |   -0.0596471 |
| readmitted        |          -0.0596471 |    1         |

admission_type_id 与 readmitted 的皮尔逊相关系数为: -0.05964710341657234

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
|  0 |      -1 | N/A          | 0.723183 | 0.897309 |     0 |     2 |        0 |      0 |    4223 |
|  1 |       0 | 1            | 0.806097 | 0.93242  |     0 |     2 |        0 |      0 |   54187 |
|  2 |       1 | 10           | 1.33333  | 1.0328   |     0 |     2 |        2 |      2 |       6 |
|  3 |       2 | 12           | 0.666667 | 0.57735  |     0 |     1 |        1 |      1 |       3 |
|  4 |       3 | 13           | 0.225434 | 0.59079  |     0 |     2 |        0 |      0 |     346 |
|  5 |       4 | 14           | 0.108504 | 0.371742 |     0 |     2 |        0 |      0 |     341 |
|  6 |       5 | 15           | 1        | 0.765641 |     0 |     2 |        1 |      1 |      59 |
|  7 |       6 | 16           | 1.11111  | 1.05409  |     0 |     2 |        2 |      2 |       9 |
|  8 |       7 | 17           | 0.666667 | 0.984732 |     0 |     2 |        0 |      0 |      12 |
|  9 |       8 | 2            | 0.798732 | 0.893457 |     0 |     2 |        0 |      0 |    1893 |
| 10 |       9 | 22           | 0.787828 | 0.822645 |     0 |     2 |        1 |      0 |    1791 |
| 11 |      10 | 23           | 0.786842 | 0.941192 |     0 |     2 |        0 |      0 |     380 |
| 12 |      11 | 24           | 0.833333 | 0.934871 |     0 |     2 |        0 |      0 |      42 |
| 13 |      12 | 27           | 0.5      | 1        |     0 |     2 |        0 |      0 |       4 |
| 14 |      13 | 28           | 0.838462 | 0.775635 |     0 |     2 |        1 |      0 |     130 |
| 15 |      14 | 3            | 0.851352 | 0.91202  |     0 |     2 |        0 |      0 |   12607 |
| 16 |      15 | 4            | 0.795393 | 0.915793 |     0 |     2 |        0 |      0 |     738 |
| 17 |      16 | 5            | 0.80859  | 0.866706 |     0 |     2 |        1 |      0 |    1071 |
| 18 |      17 | 6            | 0.962551 | 0.932562 |     0 |     2 |        1 |      0 |   11589 |
| 19 |      18 | 7            | 0.870736 | 0.920092 |     0 |     2 |        1 |      0 |     557 |
| 20 |      19 | 8            | 0.826531 | 0.930863 |     0 |     2 |        0 |      0 |      98 |
| 21 |      20 | 9            | 0.631579 | 0.683986 |     0 |     2 |        1 |      0 |      19 |

discharge_disposition_id 与 readmitted 的相关系数矩阵：

|                          |   discharge_disposition_id |   readmitted |
|:-------------------------|---------------------------:|-------------:|
| discharge_disposition_id |                  1         |    0.0502221 |
| readmitted               |                  0.0502221 |    1         |

discharge_disposition_id 与 readmitted 的皮尔逊相关系数为: 0.050222100310701574

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

|    |   class | class_name   |     mean |        std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------|---------:|-----------:|------:|------:|---------:|-------:|--------:|
|  0 |      -1 | N/A          | 0.852719 |   0.933336 |     0 |     2 |        0 |      0 |    6233 |
|  1 |       0 | 1            | 0.762326 |   0.91463  |     0 |     2 |        0 |      0 |   26347 |
|  2 |       1 | 10           | 0.5      |   0.92582  |     0 |     2 |        0 |      0 |       8 |
|  3 |       2 | 11           | 0        | nan        |     0 |     0 |        0 |      0 |       1 |
|  4 |       3 | 13           | 0        | nan        |     0 |     0 |        0 |      0 |       1 |
|  5 |       4 | 14           | 0        |   0        |     0 |     0 |        0 |      0 |       2 |
|  6 |       5 | 2            | 0.692627 |   0.89757  |     0 |     2 |        0 |      0 |     963 |
|  7 |       6 | 22           | 0.666667 |   0.887625 |     0 |     2 |        0 |      0 |      12 |
|  8 |       7 | 25           | 0        |   0        |     0 |     0 |        0 |      0 |       2 |
|  9 |       8 | 3            | 0.74269  |   0.883437 |     0 |     2 |        0 |      0 |     171 |
| 10 |       9 | 4            | 0.530869 |   0.825066 |     0 |     2 |        0 |      0 |    2786 |
| 11 |      10 | 5            | 0.695291 |   0.889891 |     0 |     2 |        0 |      0 |     722 |
| 12 |      11 | 6            | 0.444279 |   0.773214 |     0 |     2 |        0 |      0 |    2019 |
| 13 |      12 | 7            | 0.887966 |   0.931578 |     0 |     2 |        1 |      0 |   50824 |
| 14 |      13 | 8            | 0.642857 |   0.928783 |     0 |     2 |        0 |      0 |      14 |

admission_source_id 与 readmitted 的相关系数矩阵：

|                     |   admission_source_id |   readmitted |
|:--------------------|----------------------:|-------------:|
| admission_source_id |             1         |    0.0446578 |
| readmitted          |             0.0446578 |    1         |

admission_source_id 与 readmitted 的皮尔逊相关系数为: 0.044657784449454486

## time_in_hospital: (numeric, continuous)

time_in_hospital 与 readmitted 的皮尔逊相关系数为: 0.04440852358167005

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

|    |   class | class_name   |     mean |        std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------|---------:|-----------:|------:|------:|---------:|-------:|--------:|
|  0 |      -1 | N/A          | 0.817876 |   0.921611 |     0 |     2 |        0 |      0 |   35646 |
|  1 |       0 | BC           | 0.657488 |   0.888888 |     0 |     2 |        0 |      0 |    4140 |
|  2 |       1 | CH           | 0.569231 |   0.85307  |     0 |     2 |        0 |      0 |     130 |
|  3 |       2 | CM           | 0.808187 |   0.92526  |     0 |     2 |        0 |      0 |    1710 |
|  4 |       3 | CP           | 0.69849  |   0.90712  |     0 |     2 |        0 |      0 |    2252 |
|  5 |       4 | DM           | 0.932238 |   0.93488  |     0 |     2 |        1 |      0 |     487 |
|  6 |       5 | FR           | 0        | nan        |     0 |     0 |        0 |      0 |       1 |
|  7 |       6 | HM           | 0.849911 |   0.935581 |     0 |     2 |        0 |      0 |    5610 |
|  8 |       7 | MC           | 0.873182 |   0.92927  |     0 |     2 |        0 |      0 |   28529 |
|  9 |       8 | MD           | 0.839241 |   0.925926 |     0 |     2 |        0 |      0 |    3160 |
| 10 |       9 | MP           | 1.08571  |   0.94398  |     0 |     2 |        1 |      2 |      70 |
| 11 |      10 | OG           | 0.788274 |   0.907525 |     0 |     2 |        0 |      0 |     921 |
| 12 |      11 | OT           | 0.914634 |   0.971205 |     0 |     2 |        0 |      0 |      82 |
| 13 |      12 | PO           | 0.569524 |   0.861288 |     0 |     2 |        0 |      0 |     525 |
| 14 |      13 | SI           | 0.735849 |   0.901941 |     0 |     2 |        0 |      0 |      53 |
| 15 |      14 | SP           | 0.847787 |   0.934873 |     0 |     2 |        0 |      0 |    4474 |
| 16 |      15 | UN           | 0.674737 |   0.896982 |     0 |     2 |        0 |      0 |    2189 |
| 17 |      16 | WC           | 0.380952 |   0.767929 |     0 |     2 |        0 |      0 |     126 |

payer_code 与 readmitted 的相关系数矩阵：

|            |   payer_code |   readmitted |
|:-----------|-------------:|-------------:|
| payer_code |    1         |    0.0114499 |
| readmitted |    0.0114499 |    1         |

payer_code 与 readmitted 的皮尔逊相关系数为: 0.011449854375543146

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
|  0 |      -1 | N/A                                  | 0.85969  |   0.928711 |     0 |     2 |      0   |      0 |   44259 |
|  1 |       0 | AllergyandImmunology                 | 1.28571  |   0.755929 |     0 |     2 |      1   |      1 |       7 |
|  2 |       1 | Anesthesiology                       | 0.454545 |   0.8202   |     0 |     2 |      0   |      0 |      11 |
|  3 |       2 | Anesthesiology-Pediatric             | 0.611111 |   0.916444 |     0 |     2 |      0   |      0 |      18 |
|  4 |       3 | Cardiology                           | 0.784098 |   0.934234 |     0 |     2 |      0   |      0 |    4729 |
|  5 |       4 | Cardiology-Pediatric                 | 1.28571  |   0.95119  |     0 |     2 |      2   |      2 |       7 |
|  6 |       5 | DCPTEAM                              | 0.4      |   0.894427 |     0 |     2 |      0   |      0 |       5 |
|  7 |       6 | Dentistry                            | 1.5      |   1        |     0 |     2 |      2   |      2 |       4 |
|  8 |       7 | Dermatology                          | 2        | nan        |     2 |     2 |      2   |      2 |       1 |
|  9 |       8 | Emergency/Trauma                     | 0.924306 |   0.938343 |     0 |     2 |      1   |      0 |    6698 |
| 10 |       9 | Endocrinology                        | 0.72381  |   0.935365 |     0 |     2 |      0   |      0 |     105 |
| 11 |      10 | Endocrinology-Metabolism             | 0.4      |   0.894427 |     0 |     2 |      0   |      0 |       5 |
| 12 |      11 | Family/GeneralPractice               | 0.853718 |   0.92658  |     0 |     2 |      0   |      0 |    6590 |
| 13 |      12 | Gastroenterology                     | 0.899384 |   0.942885 |     0 |     2 |      1   |      0 |     487 |
| 14 |      13 | Gynecology                           | 0.333333 |   0.752355 |     0 |     2 |      0   |      0 |      54 |
| 15 |      14 | Hematology                           | 0.929577 |   0.816332 |     0 |     2 |      1   |      0 |      71 |
| 16 |      15 | Hematology/Oncology                  | 0.845304 |   0.880872 |     0 |     2 |      1   |      0 |     181 |
| 17 |      16 | Hospitalist                          | 0.72549  |   0.939754 |     0 |     2 |      0   |      0 |      51 |
| 18 |      17 | InfectiousDiseases                   | 0.941176 |   0.885615 |     0 |     2 |      1   |      0 |      34 |
| 19 |      18 | InternalMedicine                     | 0.770232 |   0.912331 |     0 |     2 |      0   |      0 |   12913 |
| 20 |      19 | Nephrology                           | 1.03401  |   0.913096 |     0 |     2 |      1   |      2 |    1382 |
| 21 |      20 | Neurology                            | 0.502762 |   0.840795 |     0 |     2 |      0   |      0 |     181 |
| 22 |      21 | Neurophysiology                      | 0        | nan        |     0 |     0 |      0   |      0 |       1 |
| 23 |      22 | Obsterics&Gynecology-GynecologicOnco | 0.5      |   0.82717  |     0 |     2 |      0   |      0 |      20 |
| 24 |      23 | Obstetrics                           | 0.277778 |   0.669113 |     0 |     2 |      0   |      0 |      18 |
| 25 |      24 | ObstetricsandGynecology              | 0.377076 |   0.749269 |     0 |     2 |      0   |      0 |     602 |
| 26 |      25 | Oncology                             | 0.820423 |   0.869266 |     0 |     2 |      1   |      0 |     284 |
| 27 |      26 | Ophthalmology                        | 0.6875   |   0.931094 |     0 |     2 |      0   |      0 |      32 |
| 28 |      27 | Orthopedics                          | 0.577674 |   0.84711  |     0 |     2 |      0   |      0 |    1281 |
| 29 |      28 | Orthopedics-Reconstructive           | 0.548329 |   0.847379 |     0 |     2 |      0   |      0 |    1107 |
| 30 |      29 | Osteopath                            | 1.0303   |   0.98377  |     0 |     2 |      1   |      2 |      33 |
| 31 |      30 | Otolaryngology                       | 0.474138 |   0.849155 |     0 |     2 |      0   |      0 |     116 |
| 32 |      31 | OutreachServices                     | 0.916667 |   0.996205 |     0 |     2 |      0.5 |      0 |      12 |
| 33 |      32 | Pathology                            | 1.35714  |   0.928783 |     0 |     2 |      2   |      2 |      14 |
| 34 |      33 | Pediatrics                           | 0.656388 |   0.905241 |     0 |     2 |      0   |      0 |     227 |
| 35 |      34 | Pediatrics-AllergyandImmunology      | 2        |   0        |     2 |     2 |      2   |      2 |       2 |
| 36 |      35 | Pediatrics-CriticalCare              | 0.64557  |   0.920395 |     0 |     2 |      0   |      0 |      79 |
| 37 |      36 | Pediatrics-EmergencyMedicine         | 0.666667 |   1.1547   |     0 |     2 |      0   |      0 |       3 |
| 38 |      37 | Pediatrics-Endocrinology             | 0.288732 |   0.700365 |     0 |     2 |      0   |      0 |     142 |
| 39 |      38 | Pediatrics-Hematology-Oncology       | 0.25     |   0.5      |     0 |     1 |      0   |      0 |       4 |
| 40 |      39 | Pediatrics-InfectiousDiseases        | 2        | nan        |     2 |     2 |      2   |      2 |       1 |
| 41 |      40 | Pediatrics-Neurology                 | 0.8      |   1.0328   |     0 |     2 |      0   |      0 |      10 |
| 42 |      41 | Pediatrics-Pulmonology               | 1.33333  |   0.912871 |     0 |     2 |      2   |      2 |      21 |
| 43 |      42 | Perinatology                         | 0        | nan        |     0 |     0 |      0   |      0 |       1 |
| 44 |      43 | PhysicalMedicineandRehabilitation    | 0.609827 |   0.831145 |     0 |     2 |      0   |      0 |     346 |
| 45 |      44 | PhysicianNotFound                    | 1        |   0.942809 |     0 |     2 |      1   |      0 |      10 |
| 46 |      45 | Podiatry                             | 1.03529  |   0.944244 |     0 |     2 |      1   |      2 |      85 |
| 47 |      46 | Proctology                           | 0        | nan        |     0 |     0 |      0   |      0 |       1 |
| 48 |      47 | Psychiatry                           | 0.736292 |   0.897878 |     0 |     2 |      0   |      0 |     766 |
| 49 |      48 | Psychiatry-Addictive                 | 0        | nan        |     0 |     0 |      0   |      0 |       1 |
| 50 |      49 | Psychiatry-Child/Adolescent          | 0.714286 |   0.95119  |     0 |     2 |      0   |      0 |       7 |
| 51 |      50 | Psychology                           | 0.795699 |   0.950505 |     0 |     2 |      0   |      0 |      93 |
| 52 |      51 | Pulmonology                          | 0.881579 |   0.934863 |     0 |     2 |      0   |      0 |     760 |
| 53 |      52 | Radiologist                          | 0.762327 |   0.925519 |     0 |     2 |      0   |      0 |    1014 |
| 54 |      53 | Radiology                            | 0.723404 |   0.877301 |     0 |     2 |      0   |      0 |      47 |
| 55 |      54 | Resident                             | 1.5      |   0.707107 |     1 |     2 |      1.5 |      1 |       2 |
| 56 |      55 | Rheumatology                         | 0.75     |   0.930949 |     0 |     2 |      0   |      0 |      16 |
| 57 |      56 | Speech                               | 0        | nan        |     0 |     0 |      0   |      0 |       1 |
| 58 |      57 | SportsMedicine                       | 2        | nan        |     2 |     2 |      2   |      2 |       1 |
| 59 |      58 | Surgeon                              | 0.45     |   0.782829 |     0 |     2 |      0   |      0 |      40 |
| 60 |      59 | Surgery-Cardiovascular               | 0.541176 |   0.852907 |     0 |     2 |      0   |      0 |      85 |
| 61 |      60 | Surgery-Cardiovascular/Thoracic      | 0.45679  |   0.800498 |     0 |     2 |      0   |      0 |     567 |
| 62 |      61 | Surgery-Colon&Rectal                 | 0.5      |   0.849837 |     0 |     2 |      0   |      0 |      10 |
| 63 |      62 | Surgery-General                      | 0.795356 |   0.918568 |     0 |     2 |      0   |      0 |    2756 |
| 64 |      63 | Surgery-Maxillofacial                | 0.5      |   0.849837 |     0 |     2 |      0   |      0 |      10 |
| 65 |      64 | Surgery-Neuro                        | 0.400474 |   0.760235 |     0 |     2 |      0   |      0 |     422 |
| 66 |      65 | Surgery-Pediatric                    | 0.5      |   0.92582  |     0 |     2 |      0   |      0 |       8 |
| 67 |      66 | Surgery-Plastic                      | 0.675    |   0.858965 |     0 |     2 |      0   |      0 |      40 |
| 68 |      67 | Surgery-PlasticwithinHeadandNeck     | 2        | nan        |     2 |     2 |      2   |      2 |       1 |
| 69 |      68 | Surgery-Thoracic                     | 0.649485 |   0.87846  |     0 |     2 |      0   |      0 |      97 |
| 70 |      69 | Surgery-Vascular                     | 0.894958 |   0.923093 |     0 |     2 |      1   |      0 |     476 |
| 71 |      70 | SurgicalSpecialty                    | 0.5      |   0.83887  |     0 |     2 |      0   |      0 |      28 |
| 72 |      71 | Urology                              | 0.640523 |   0.881632 |     0 |     2 |      0   |      0 |     612 |

medical_specialty 与 readmitted 的相关系数矩阵：

|                   |   medical_specialty |   readmitted |
|:------------------|--------------------:|-------------:|
| medical_specialty |           1         |   -0.0531533 |
| readmitted        |          -0.0531533 |    1         |

medical_specialty 与 readmitted 的皮尔逊相关系数为: -0.053153337586684224

## num_lab_procedures: (numeric, continuous)

num_lab_procedures 与 readmitted 的皮尔逊相关系数为: 0.042350422253684836

## num_procedures: (numeric, continuous)

num_procedures 与 readmitted 的皮尔逊相关系数为: -0.039416441636573256

## num_medications: (numeric, continuous)

num_medications 与 readmitted 的皮尔逊相关系数为: 0.042785202699440195

## number_outpatient: (numeric, continuous)

number_outpatient 与 readmitted 的皮尔逊相关系数为: 0.08404365343913692

## number_emergency: (numeric, continuous)

number_emergency 与 readmitted 的皮尔逊相关系数为: 0.09010913534282058

## number_inpatient: (numeric, continuous)

number_inpatient 与 readmitted 的皮尔逊相关系数为: 0.1814228545783148

## number_diagnoses: (numeric, continuous)

number_diagnoses 与 readmitted 的皮尔逊相关系数为: 0.11365743451407587

## max_glu_serum: (categorical, categorical)

max_glu_serum
-1.0    85420
 2.0     2280
 0.0     1305
 1.0     1100
Name: count, dtype: int64

{0: '>200', 1: '>300', 2: 'Norm'}

### statics

|    |   class | class_name   |     mean |      std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------|---------:|---------:|------:|------:|---------:|-------:|--------:|
|  0 |      -1 | N/A          | 0.820733 | 0.924603 |     0 |     2 |        0 |      0 |   85420 |
|  1 |       0 | >200         | 0.878161 | 0.924955 |     0 |     2 |        1 |      0 |    1305 |
|  2 |       1 | >300         | 1.00636  | 0.922352 |     0 |     2 |        1 |      2 |    1100 |
|  3 |       2 | Norm         | 0.817105 | 0.923757 |     0 |     2 |        0 |      0 |    2280 |

max_glu_serum 与 readmitted 的相关系数矩阵：

|               |   max_glu_serum |   readmitted |
|:--------------|----------------:|-------------:|
| max_glu_serum |      1          |   0.00973073 |
| readmitted    |      0.00973073 |   1          |

max_glu_serum 与 readmitted 的皮尔逊相关系数为: 0.00973072754120879

## A1Cresult: (categorical, categorical)

A1Cresult
-1.0    74928
 1.0     7339
 2.0     4434
 0.0     3404
Name: count, dtype: int64

{0: '>7', 1: '>8', 2: 'Norm'}

### statics

|    |   class | class_name   |     mean |      std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------|---------:|---------:|------:|------:|---------:|-------:|--------:|
|  0 |      -1 | N/A          | 0.831225 | 0.924713 |     0 |     2 |        0 |      0 |   74928 |
|  1 |       0 | >7           | 0.795535 | 0.925953 |     0 |     2 |        0 |      0 |    3404 |
|  2 |       1 | >8           | 0.804878 | 0.928383 |     0 |     2 |        0 |      0 |    7339 |
|  3 |       2 | Norm         | 0.750113 | 0.915457 |     0 |     2 |        0 |      0 |    4434 |

A1Cresult 与 readmitted 的相关系数矩阵：

|            |   A1Cresult |   readmitted |
|:-----------|------------:|-------------:|
| A1Cresult  |    1        |    -0.019605 |
| readmitted |   -0.019605 |     1        |

A1Cresult 与 readmitted 的皮尔逊相关系数为: -0.01960502080105271

## metformin: (medication, categorical)

metformin 与 readmitted 的皮尔逊相关系数为: -0.027686685456782805

## repaglinide: (medication, categorical)

repaglinide 与 readmitted 的皮尔逊相关系数为: 0.017917361553174485

## nateglinide: (medication, categorical)

nateglinide 与 readmitted 的皮尔逊相关系数为: 0.0028733815979235513

## chlorpropamide: (medication, categorical)

chlorpropamide 与 readmitted 的皮尔逊相关系数为: 0.004084601060759715

## glimepiride: (medication, categorical)

glimepiride 与 readmitted 的皮尔逊相关系数为: 0.0024481053376232366

## acetohexamide: (medication, categorical)

acetohexamide 与 readmitted 的皮尔逊相关系数为: 0.004237306335417519

## glipizide: (medication, categorical)

glipizide 与 readmitted 的皮尔逊相关系数为: 0.014348748775769773

## glyburide: (medication, categorical)

glyburide 与 readmitted 的皮尔逊相关系数为: -0.006097183322941256

## tolbutamide: (medication, categorical)

tolbutamide 与 readmitted 的皮尔逊相关系数为: -0.001551676135997375

## pioglitazone: (medication, categorical)

pioglitazone 与 readmitted 的皮尔逊相关系数为: 0.010418839133825327

## rosiglitazone: (medication, categorical)

rosiglitazone 与 readmitted 的皮尔逊相关系数为: 0.013418922692326933

## acarbose: (medication, categorical)

acarbose 与 readmitted 的皮尔逊相关系数为: 0.01722159978455592

## miglitol: (medication, categorical)

miglitol 与 readmitted 的皮尔逊相关系数为: 0.0024722706932359894

## troglitazone: (medication, categorical)

troglitazone 与 readmitted 的皮尔逊相关系数为: 0.003179627290362604

## tolazamide: (medication, categorical)

tolazamide 与 readmitted 的皮尔逊相关系数为: -0.00537820413149414

## examide: (medication, categorical)

examide 与 readmitted 的皮尔逊相关系数为: nan

## citoglipton: (medication, categorical)

citoglipton 与 readmitted 的皮尔逊相关系数为: nan

## insulin: (medication, categorical)

insulin 与 readmitted 的皮尔逊相关系数为: 0.0041360109955961314

## glyburide-metformin: (medication, categorical)

glyburide-metformin 与 readmitted 的皮尔逊相关系数为: 0.0006518186615473238

## glipizide-metformin: (medication, categorical)

glipizide-metformin 与 readmitted 的皮尔逊相关系数为: 0.004287871142177058

## glimepiride-pioglitazone: (medication, categorical)

glimepiride-pioglitazone 与 readmitted 的皮尔逊相关系数为: nan

## metformin-rosiglitazone: (medication, categorical)

metformin-rosiglitazone 与 readmitted 的皮尔逊相关系数为: -0.0029673977764603215

## metformin-pioglitazone: (medication, categorical)

metformin-pioglitazone 与 readmitted 的皮尔逊相关系数为: -0.002967397776460321

## change: (categorical, categorical)

change
1    48256
0    41849
Name: count, dtype: int64

{0: 'Ch', 1: 'No'}

### statics

|    |   class | class_name   |     mean |      std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------|---------:|---------:|------:|------:|---------:|-------:|--------:|
|  0 |       0 | Ch           | 0.862912 | 0.928022 |     0 |     2 |        0 |      0 |   41849 |
|  1 |       1 | No           | 0.789767 | 0.920643 |     0 |     2 |        0 |      0 |   48256 |

change 与 readmitted 的相关系数矩阵：

|            |     change |   readmitted |
|:-----------|-----------:|-------------:|
| change     |  1         |   -0.0394467 |
| readmitted | -0.0394467 |    1         |

change 与 readmitted 的皮尔逊相关系数为: -0.03944673729560536

## diabetesMed: (categorical, categorical)

diabetesMed
1    69574
0    20531
Name: count, dtype: int64

{0: 'No', 1: 'Yes'}

### statics

|    |   class | class_name   |     mean |      std |   min |   max |   median |   mode |   count |
|---:|--------:|:-------------|---------:|---------:|------:|------:|---------:|-------:|--------:|
|  0 |       0 | No           | 0.731479 | 0.910903 |     0 |     2 |        0 |      0 |   20531 |
|  1 |       1 | Yes          | 0.850964 | 0.927105 |     0 |     2 |        0 |      0 |   69574 |

diabetesMed 与 readmitted 的相关系数矩阵：

|             |   diabetesMed |   readmitted |
|:------------|--------------:|-------------:|
| diabetesMed |     1         |    0.0541941 |
| readmitted  |     0.0541941 |    1         |

diabetesMed 与 readmitted 的皮尔逊相关系数为: 0.054194099315486056

## readmitted: (target, categorical)

readmitted 与 readmitted 的皮尔逊相关系数为:             readmitted  readmitted
readmitted         1.0         1.0
readmitted         1.0         1.0

## diag_1: (diagnosis, categorical)

diag_1 与 readmitted 的皮尔逊相关系数为: -0.04083280099607248

## diag_2: (diagnosis, categorical)

diag_2 与 readmitted 的皮尔逊相关系数为: -0.014426835168271514

## diag_3: (diagnosis, categorical)

diag_3 与 readmitted 的皮尔逊相关系数为: 0.00628249473573884

