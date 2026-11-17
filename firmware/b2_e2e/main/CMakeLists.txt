idf_component_register(
  SRCS "main.cc" "generated/model_data.cc" "generated/test_data.cc" "generated/e2e_data.cc"
  INCLUDE_DIRS "." "generated"
  PRIV_REQUIRES esp_timer esp_hw_support esp_pm)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
