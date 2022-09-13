cmake_minimum_required(VERSION 3.16)

aux_source_directory(. FILE_SOURCES)
add_library(filter ${FILE_SOURCES})

