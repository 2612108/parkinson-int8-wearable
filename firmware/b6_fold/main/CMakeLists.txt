idf_component_register(
  SRCS "main.cc" "generated/model_data.cc"
  INCLUDE_DIRS "." "generated"
  PRIV_REQUIRES esp_timer esp_driver_usb_serial_jtag)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2)
