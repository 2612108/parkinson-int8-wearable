# Audit Monipar

## Zenodo
- Tiêu đề: Monipar Database: smartwatch movement data to monitor motor competency in subjects with Parkinson's disease
- Ngày: 2023-07-02
- License: {'id': 'cc-by-4.0'}
- Phiên bản: 1.0
- File: Monipar database organization week:subject.png (0.0 MB), MONIPAR_HEALTHYCONTROL.mat (11.2 MB), MONIPAR_PD_REMOTE.mat (15.0 MB), Fast_watch_Signals.m (0.0 MB), MONIPAR_PD_REMOTE_TREMOR_LABEL.mat (0.0 MB), MONIPAR_PD_SUPERVISED_TREMOR_LABEL.mat (0.0 MB), MONIPAR_PD_SUPERVISED.mat (9.2 MB), Monipar_README.txt (0.0 MB), MONIPAR SUBJECTS DATA.xlsx (0.0 MB)

Mô tả:
```
 Movement data was collected through smartwatches to monitor motor competence in subjects with Parkinson's Disease (PD). The data set collected for the Monipar study consists of triaxial acceleration data from 21 subjects with PD and 7 healthy control subjects when performing a set of physical exercises while wearing an off-the-shelf smartwatch. Each participant performed the complete set of eight exercises once a week, commonly on the same day and at a similar time. Three Matlab files are provided that contain the raw data of the experimental subgroups: (1) Supervised, (2) Remote, and (3) Healthy control. Additionally, two Matlab files are provided containing the Tremor Labels for selected subjects in the experimental subgroups: (1) Supervised and (2) Remote. While the implementation of the experimental protocol for collecting movement data followed a consistent approach for all participants, three distinct experimental subgroups were established: Remote group: This subgroup consisted of individuals diagnosed with Parkinson's disease (PD) who completed the experimental protocol at their regular PD association. Supervised group: This subgroup comprised PD patients who underwent the experimental protocol under circumstances similar to the remote group. Additionally, clinical scoring (MDS-UPDRS) is reported for this group in the file "MONIPAR SUBJECTS DATA.xlsx" Healthy control group: This subgroup consisted of healthy participants who performed exercises under the supervision of research project team members. Data was collected using a sample rate of 50Hz and expressed in m/s^2. Check the "Monipar_README.txt" file for details about this dataset. Further details are contained in the following reference -- if you use this dataset, please cite: Sigcha, L., Polvorinos-Fernández, C., Costa, N., Costa, S., Arezes, P., Gago, M., ... & Pavón, I. " Monipar: Movement data collection tool to monitor motor symptoms in Parkinson's disease using smartwatches and smartphones ". Frontiers in Neurology , 14 , 1326640. https://doi.org/10.3389/fneur.2023.1326640 References: Sigcha, L. et al. (2022). Bradykinesia Detection in Parkinson's Disease Using Smartwatches' Inertial Sensors and Deep Learning Methods. Sensors 11, 3879 Sigcha, L. et al. (2021). Automatic Resting Tremor Assessment in Parkinson's Disease Using Smartwatches and Multitask Convolutional Neural Networks. Sensors 21, 291. Funding: This research was funded by the following projects: (1) "Tecnologías Capacitadoras para la Asistencia, Seguimiento y Rehabilitación de Pacientes con Enfermedad de Parkinson". Centro Internacional sobre el envejecimiento, CENIE (código 0348_CIE_6_E) Interreg V-A España-Portugal (POCTEP). (2) FCT—Fundação para a Ciência e Tecnologia within the R&D Units Project Scope: UIDB/00319/2020. 
```

## Kho file
- Tổng file: 10, dung lượng 35.4 MB
- Theo phần mở rộng: `.mat`: 5, `.json`: 1, `.png`: 1, `.m`: 1, `.txt`: 1, `.xlsx`: 1
- Thư mục (top 15 theo số file):
  - `.`: 10

## Danh sách file dữ liệu (7), 80 file đầu
```
MONIPAR_PD_REMOTE.mat
MONIPAR_PD_SUPERVISED.mat
MONIPAR_PD_REMOTE_TREMOR_LABEL.mat
MONIPAR_HEALTHYCONTROL.mat
MONIPAR_PD_SUPERVISED_TREMOR_LABEL.mat
Monipar_README.txt
MONIPAR SUBJECTS DATA.xlsx
```
- Token hay gặp trong tên file: monipar(7), pd(4), remote(2), supervised(2), tremor(2), label(2), healthycontrol(1), readme(1), subjects(1), data(1)

## Mẫu nội dung (10 file)
### `MONIPAR_PD_REMOTE.mat`
(file .mat — cần scipy.io.loadmat, gửi tên biến cho Claude)
### `MONIPAR_PD_SUPERVISED.mat`
(file .mat — cần scipy.io.loadmat, gửi tên biến cho Claude)
### `MONIPAR_PD_REMOTE_TREMOR_LABEL.mat`
(file .mat — cần scipy.io.loadmat, gửi tên biến cho Claude)
### `MONIPAR_HEALTHYCONTROL.mat`
(file .mat — cần scipy.io.loadmat, gửi tên biến cho Claude)
### `MONIPAR_PD_SUPERVISED_TREMOR_LABEL.mat`
(file .mat — cần scipy.io.loadmat, gửi tên biến cho Claude)
### `Monipar_README.txt`
```
MONIPAR DATABASE v1.0

The data set collected during the experimental campaign consists of triaxial-acceleration data of 21 patients with PD and 7 healthy control subjects. Each participant performed the complete set of 8 exercises once a week, commonly on the same day and at a similar hour.
Three Matlab files are provided containing the raw data of the experimental subgroups: (1) Supervised, (2) Remote, and (3) Healthy control.
Each Matlab file (*.m) contains a Data structure in which each "cell" represents a "single trial". The rows represent a different subject and the columns represent a different week.

```
### `MONIPAR SUBJECTS DATA.xlsx`
lỗi đọc: `Import openpyxl` failed.  Use pip, conda, or your preferred package management tool to install the openpyxl package.
