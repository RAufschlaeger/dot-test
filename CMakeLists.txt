cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
PROJECT(app)

find_package(Torch REQUIRED)
FIND_PACKAGE( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
INCLUDE_DIRECTORIES( ${Torch_INCLUDE_DIRS} )

ADD_EXECUTABLE(app ../src/main.cpp)
TARGET_LINK_LIBRARIES (app ${OpenCV_LIBS})
target_link_libraries(app "${TORCH_LIBRARIES}")

set_property(TARGET app PROPERTY CXX_STANDARD 14)
