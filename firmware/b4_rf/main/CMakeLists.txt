file(GLOB GEN_SRCS "generated/*.cc" "generated/*.c")
idf_component_register(SRCS "main.cc" ${GEN_SRCS}
                       INCLUDE_DIRS "." "generated"
                       REQUIRES esp_timer)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -Wno-unused-function)
