#pragma once
#include <stdint.h>

#define E2E_N_WIN 8
#define E2E_RAW_C 6
#define E2E_RAW_T 400
#define E2E_DECIM_T 200
#define E2E_OUT_C 8
#define E2E_FIR_LEN 41
#define E2E_SOS_SECTIONS 4

extern const char* const g_e2e_win_desc[E2E_N_WIN];
extern const char* const g_e2e_win_file[E2E_N_WIN];
extern const int g_e2e_win_start[E2E_N_WIN];
extern const char* const g_e2e_win_label[E2E_N_WIN];

// Tín hiệu thô 100 Hz: [E2E_N_WIN][6][400]
extern const float g_e2e_raw_in[E2E_N_WIN][E2E_RAW_C][E2E_RAW_T];

// Hệ số FIR polyphase hạ mẫu 100 -> 50 Hz (41 taps)
extern const float g_fir_kaiser41[E2E_FIR_LEN];

// Hệ số IIR Butterworth thông dải 0.5-20 Hz bậc 4 nhân quả @ 50 Hz: [4][6] = {b0, b1, b2, a0, a1, a2}
extern const float g_butter4_bp_sos[E2E_SOS_SECTIONS][6];

// Tham số chuẩn hóa z-score (8 kênh: ax, ay, az, wx, wy, wz, |a|, |w|)
extern const float g_norm_mu[E2E_OUT_C];
extern const float g_norm_sd[E2E_OUT_C];

// Đầu ra tham chiếu trên PC cho từng cửa sổ
extern const int8_t g_ref_q_causal[E2E_N_WIN];
extern const float g_ref_prob_causal[E2E_N_WIN];
extern const int8_t g_ref_q_zerophase_win[E2E_N_WIN];
extern const float g_ref_prob_zerophase_win[E2E_N_WIN];
extern const int8_t g_ref_q_zerophase_full[E2E_N_WIN];
extern const float g_ref_prob_zerophase_full[E2E_N_WIN];
extern const int32_t g_ref_qin_checksum[E2E_N_WIN];
