# Audit PADS

## Kho file
- Tổng file: 12700, dung lượng 1519.4 MB
- Theo phần mở rộng: `.txt`: 10320, `.json`: 1407, `.bin`: 938, `.py`: 18, `.html`: 10, `.jpg`: 6, `.csv`: 1
- Thư mục (top 15 theo số file):
  - `movement/timeseries`: 10319
  - `patients`: 470
  - `questionnaire`: 470
  - `movement`: 470
  - `preprocessed/questionnaire`: 470
  - `preprocessed/movement`: 470
  - `scripts`: 20
  - `scripts/utils`: 6
  - `.`: 3
  - `preprocessed`: 2

## Bệnh nhân (469 file JSON)
- Cột: resource_type, id, study_id, condition, disease_comment, age_at_diagnosis, age, height, weight, gender, handedness, appearance_in_kinship, appearance_in_first_grade_kinship, effect_of_alcohol_on_tremor
- `condition`: Parkinson's=276, Healthy=79, Other Movement Disorders=60, Essential Tremor=28, Atypical Parkinsonism=15, Multiple Sclerosis=11
- `gender`: male=281, female=188
- `handedness`: right=437, left=32
- `age`: mean 64.2 ± 10.7, thiếu 0
- `age_at_diagnosis`: mean 55.3 ± 17.1, thiếu 0
- `height`: mean 174.2 ± 10.9, thiếu 0
- `weight`: mean 81.9 ± 17.9, thiếu 0
- Đã ghi `reports/pads_patients.csv`

## `preprocessed/file_list.csv`
- shape (469, 15); cột: resource_type, id, study_id, condition, disease_comment, age_at_diagnosis, age, height, weight, gender, handedness, appearance_in_kinship, appearance_in_first_grade_kinship, effect_of_alcohol_on_tremor, label
```
  resource_type  id study_id                 condition                                                                 disease_comment  age_at_diagnosis  age  height  weight  gender handedness  appearance_in_kinship appearance_in_first_grade_kinship effect_of_alcohol_on_tremor  label
0       patient   1     PADS                   Healthy                                                                               -                56   56     173      78    male      right                   True                              True                     Unknown      0
1       patient   2     PADS  Other Movement Disorders  Left-Sided resting tremor and hypokinesia without presynaptic striatal deficit                69   81     193     104    male      right                  False                               NaN                   No effect      2
2       patient   3     PADS                   Healthy                                                                               -                45   45     170      78  female      right                  False                               NaN                     Unknown      0
3       patient   4     PADS               Parkinson's                                                         IPS akinetic-rigid type                63   67     161      90  female      right                  False                               NaN                   No effect      1
4       patient   5     PADS               Parkinson's                                                         IPS tremordominant type                65   75     172      86    male       left                  False                               NaN                     Unknown      1
```
- `resource_type`: patient=469
- `study_id`: PADS=469
- `condition`: Parkinson's=276, Healthy=79, Other Movement Disorders=60, Essential Tremor=28, Atypical Parkinsonism=15, Multiple Sclerosis=11
- `gender`: male=281, female=188
- `handedness`: right=437, left=32
- `appearance_in_kinship`: False=343, True=126
- `appearance_in_first_grade_kinship`: False=91, True=90
- `effect_of_alcohol_on_tremor`: Unknown=228, No effect=181, Improvement=34, Worsening=26
- `label`: 1=276, 2=114, 0=79

## Chuỗi thời gian thô (10318 file)
- Số người có tín hiệu: 469
- Bài tập: TouchIndex (938), HoldWeight (938), LiftHold (938), StretchHold (938), Entrainment (938), DrinkGlas (938), RelaxedTask (938), PointFinger (938), TouchNose (938), CrossArms (938), Relaxed (938)
- Cổ tay: RightWrist=5159, LeftWrist=5159
- File/người: min 22, median 22, max 22

Mẫu 30 file (cột 0 giả định là thời gian, 1–3 acc, 4–6 gyro — **cần anh xác nhận với readme của PADS**):
```
                          file    n  cols     fs t_unit  jitter  acc_absmax  gyro_absmax
465_StretchHold_RightWrist.txt 1023     7  99.34      s  0.0229    0.205875     0.425106
     194_Relaxed_LeftWrist.txt 2047     7 100.05      s  0.0605    0.204618     0.537298
   246_DrinkGlas_LeftWrist.txt 1023     7 100.07      s  0.0956    1.949328     7.177814
200_RelaxedTask_RightWrist.txt 2047     7  99.33      s  0.0282    0.053091     0.470913
 393_Entrainment_LeftWrist.txt 2047     7 100.05      s  0.0371    0.141511     0.507945
 413_StretchHold_LeftWrist.txt 1023     7  99.57      s  0.0278    0.223209     0.343199
 210_TouchIndex_RightWrist.txt 1023     7 100.80      s  0.4289    1.004689     4.572082
  164_CrossArms_RightWrist.txt 1023     7 100.74      s  0.0547    0.150880     0.790565
 398_HoldWeight_RightWrist.txt 1023     7  99.33      s  0.0278    0.217554     0.309676
     140_Relaxed_LeftWrist.txt 2047     7 100.09      s  0.0872    0.092016     1.323479
018_StretchHold_RightWrist.txt 1023     7  99.34      s  0.0260    0.175415     0.780334
     366_Relaxed_LeftWrist.txt 2047     7 100.05      s  0.0238    0.173253     0.558605
 386_PointFinger_LeftWrist.txt 1023     7  99.52      s  0.0276    1.058550     6.464923
 206_RelaxedTask_LeftWrist.txt 2047     7 100.09      s  0.0957    0.124268     0.510776
  405_HoldWeight_LeftWrist.txt 1023     7 100.04      s  0.0332    0.206477     0.442160
  069_CrossArms_RightWrist.txt 1023     7  99.33      s  0.0831    0.593669     3.125117
  468_HoldWeight_LeftWrist.txt 1023     7  99.53      s  0.0195    0.149958     0.858768
   224_DrinkGlas_LeftWrist.txt 1023     7 100.05      s  0.1640    1.401510     4.053395
 429_StretchHold_LeftWrist.txt 1023     7 100.07      s  0.1358    0.089669     0.413582
 088_TouchIndex_RightWrist.txt 1023     7  99.33      s  0.0241    0.407700     1.688722
   200_CrossArms_LeftWrist.txt 1023     7 100.04      s  0.0847    0.836139     4.635231
   211_DrinkGlas_LeftWrist.txt 1023     7 100.06      s  0.0234    1.194947     3.588994
    155_Relaxed_RightWrist.txt 2047     7  99.33      s  0.0220    0.057478     0.372730
157_PointFinger_RightWrist.txt 1023     7  99.34      s  0.0306    0.184222     0.552538
  190_TouchNose_RightWrist.txt 1023     7 100.75      s  0.0192    1.792308     4.558372
 385_Entrainment_LeftWrist.txt 2047     7 100.07      s  0.1114    0.234383     0.216760
   222_TouchNose_LeftWrist.txt 1023     7 100.06      s  0.0233    1.567755     4.820121
 006_TouchIndex_RightWrist.txt 1023     7 100.75      s  0.0257    0.186545     1.240022
    253_LiftHold_LeftWrist.txt 1023     7 100.07      s  0.0957    0.275358     1.299889
411_PointFinger_RightWrist.txt 1023     7  99.35      s  0.0148    0.794097     2.537525
```

5 dòng đầu `149_TouchIndex_RightWrist.txt`:
```
0.0000000000,0.1130211055,0.2199303508,0.4314695299,0.7900941372,2.7676019669,-1.8772745132
0.0114480257,0.1035784185,0.2336963862,0.4639118612,0.7539113760,2.7553853989,-1.9123108387
0.0197789669,0.0952951908,0.2395731360,0.5083031654,0.6789188385,2.7821044922,-1.9451942444
0.0298240185,0.0929888487,0.2392415404,0.5304511189,0.6074324846,2.7803120613,-1.9611009359
0.0402430296,0.0879943073,0.2718175650,0.5508072972,0.5623739958,2.8032357693,-1.9653260708

```

## Observation JSON (469 file) — ví dụ `observation_001.json`
```json
{
 "resource_type": "observation",
 "subject_id": "001",
 "study_id": "PADS",
 "device_id": "Apple Watch Series 4",
 "id": "Neurological Assessment",
 "endianness": "little",
 "sampling_rate": 100,
 "data_type": "float",
 "bits": 32,
 "session": [
  {
   "record_name": "Relaxed",
   "rows": 2048,
   "records": [
    {
     "device_location": "LeftWrist",
     "channels": [
      "Time",
      "Accelerometer_X",
      "Accelerometer_Y",
      "Accelerometer_Z",
      "Gyroscope_X",
      "Gyroscope_Y",
      "Gyroscope_Z"
     ],
     "units": [
      "s",
      "g",
      "g",
      "g",
      "rad/s",
      "rad/s",
      "rad/s"
     ],
     "file_name": "timeseries/001_Relaxed_LeftWrist.txt"
    },
    {
     "device_location": "RightWrist",
     "channels": [
      "Time",
      "Accelerometer_X",
      "Accelerometer_Y",
      "Accelerometer_Z",
      "Gyroscope_X",
      "Gyroscope_Y",
      "Gyroscope_Z"
     ],
     "units": [
      "s",
      "g",
      "g",
      "g",
      "rad/s",
      "rad/s",
      "rad/s"
     ],
     "file_name": "timeseries/001_Relaxed_RightWrist.txt"
    }
   ]
  },
  {
   "record_name": "RelaxedTask",
   "rows": 2048,
   "records": [
    {
     "device_location": "LeftWrist",
     "channels": [
      "Time",
      "Accelerometer_X",
      "Accelerometer_Y",
      "Accelerometer_Z",
      "Gyroscope_X",
      "Gyroscope_Y",
      "Gyroscope_Z"
     ],
     "units": [
      "s",
      "g",
      "g",
      "g",
      "rad/s",
      "rad/s",
      "rad/s"
     ],
     "file_name": "timeseries/001_RelaxedTask_LeftWrist.txt"
    },
    {
     "device_location": "RightWrist",
     "channels": [
      "Time",
      "Accelerometer_X",
      "Accelerometer_Y",
      "Accelerometer_Z",
      "Gyroscope_X",
      "Gyroscope_Y",
      "Gyroscope_Z"
     ],
     "units": [
      "s",
      "g",
      "g",
      "g",
      "rad/s",
      "rad/s",
      "rad/s"
     ],
     "file_name": "timeseries/001_RelaxedTask_RightWrist.txt"
    }
   ]
  },
  {
   "record_name": "StretchHold",
   "rows": 1024,
   "records": [
    {
     "device_location": "LeftWrist",
     "channels": [
      "Time",
      "Accelerometer_X",
      "Accelerometer_Y",
      "Accelerometer_Z",
      "Gyroscope_X",
      "Gyroscope_Y",
      "Gyroscope_Z"
     ],
     "units": [
      "s",
      "g",
      "g",
      "g",
      "rad/s",
      "rad/s",
      "rad/s"
     ],
     "file_name": "timeseries/001_StretchHold_LeftWrist.txt"
    },
    {
     "device_location
```
