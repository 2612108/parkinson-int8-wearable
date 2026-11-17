#pragma once
#include <stdint.h>

#define B4_N_TEST 20
#define B4_N_FEAT 2112
#define B4_N_TREES 100
#define B4_MAX_DEPTH 8

#define B4_REC_RAW_C 6
#define B4_REC_RAW_T 1000
#define B4_REC_DECIM_T 500
#define B4_REC_OUT_C 8
#define B4_FEATS_PER_CH 12
#define B4_REC_N_FEATS 96
#define B4_FIR_LEN 41
#define B4_SOS_SECTIONS 4

extern const char* const g_b4_test_sid[B4_N_TEST];
extern const int32_t g_b4_test_y_true[B4_N_TEST];
extern const int32_t g_b4_test_y_sklearn[B4_N_TEST];
extern const float g_b4_test_prob_sklearn[B4_N_TEST];
extern const float g_b4_test_prob_eml_soft[B4_N_TEST];
extern const float g_b4_test_prob_eml_hard[B4_N_TEST];
extern const float g_b4_test_X[B4_N_TEST][B4_N_FEAT];

extern const char* const g_b4_rec_file;
extern const float g_b4_rec_raw_10s[B4_REC_RAW_C][B4_REC_RAW_T];
extern const float g_b4_fir_kaiser41[B4_FIR_LEN];
extern const float g_b4_butter4_bp_sos[B4_SOS_SECTIONS][6];
extern const float g_b4_ref_feats96_causal[B4_REC_N_FEATS];
extern const float g_b4_ref_feats96_zp[B4_REC_N_FEATS];
extern const char* const g_b4_feat96_names[B4_REC_N_FEATS];
