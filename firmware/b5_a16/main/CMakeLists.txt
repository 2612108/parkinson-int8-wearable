idf_component_register(
  SRCS "main.cc" "generated/model_data.cc" "generated/test_data.cc"
  INCLUDE_DIRS "." "generated"
  PRIV_REQUIRES esp_timer)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
