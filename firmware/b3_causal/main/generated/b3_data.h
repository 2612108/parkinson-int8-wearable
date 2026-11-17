#pragma once
#include <stdint.h>

#define B3_N_REC 3
#define B3_RAW_C 6
#define B3_RAW_TOTAL_T 2046
#define B3_BLOCK_RAW_T 200
#define B3_BLOCK_DECIM_T 100
#define B3_N_BLOCKS 10
#define B3_WIN_DECIM_T 200
#define B3_HOP_DECIM_T 100
#define B3_N_WIN 9
#define B3_OUT_C 8
#define B3_FIR_LEN 41
#define B3_SOS_SECTIONS 4

extern const char* const g_b3_rec_file[B3_N_REC];
extern const char* const g_b3_rec_label[B3_N_REC];
extern const char* const g_b3_rec_desc[B3_N_REC];

// Toàn bộ bản ghi PADS thật 100 Hz: [B3_N_REC][6][2046]
extern const float g_b3_raw_records[B3_N_REC][B3_RAW_C][B3_RAW_TOTAL_T];

// Hệ số FIR polyphase hạ mẫu 100 -> 50 Hz (41 taps Kaiser beta=5.0)
extern const float g_fir_kaiser41[B3_FIR_LEN];

// Hệ số IIR Butterworth thông dải 0.5-20 Hz bậc 4 nhân quả @ 50 Hz: [4][6]
extern const float g_butter4_bp_sos[B3_SOS_SECTIONS][6];

// Tham số chuẩn hóa z-score (reports/p56k0/models/norm_PD_vs_HC_raw_fold0.npz)
extern const float g_norm_mu[B3_OUT_C];
extern const float g_norm_sd[B3_OUT_C];

// Đầu ra tham chiếu của đường ống Python chuẩn (resample_poly cả bản ghi + sosfilt cả bản ghi + cắt cửa sổ 200 hop 100)
extern const int8_t g_ref_q_py_cont[B3_N_REC][B3_N_WIN];
extern const float g_ref_prob_py_cont[B3_N_REC][B3_N_WIN];
extern const int32_t g_ref_qin_cksum[B3_N_REC][B3_N_WIN];

// Đầu ra tham chiếu bổ sung (zero-phase cả bản ghi và nhân quả reset mỗi cửa sổ)
extern const int8_t g_ref_q_py_zp[B3_N_REC][B3_N_WIN];
extern const float g_ref_prob_py_zp[B3_N_REC][B3_N_WIN];
extern const int8_t g_ref_q_py_reset[B3_N_REC][B3_N_WIN];
extern const float g_ref_prob_py_reset[B3_N_REC][B3_N_WIN];
