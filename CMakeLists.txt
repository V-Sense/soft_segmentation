cmake_minimum_required(VERSION 2.8)

set(PROJECTNAME "SoftSegmentation")
project(${PROJECTNAME} C CXX)

# for gdb debug -> run cmake -DCMAKE_BUILD_TYPE=Debug <other params>
set(CMAKE_BUILD_TYPE Debug) #Debug or Release

# Enable C++11 features
if(CMAKE_COMPILER_IS_GNUCXX)   
	set(CMAKE_CXX_FLAGS "-std=gnu++11") 
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")   
       set(CMAKE_CXX_FLAGS "-std=c++11") 
endif()

find_package(OpenCV REQUIRED)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

find_package(Threads REQUIRED)

include_directories(src)
include_directories(include)
include_directories(include/CPM_include)

#file(GLOB HEADERS include/*.h)
file(GLOB SOURCES src/*.cpp)

#message(STATUS ${CMAKE_SOURCE_DIR})

#create a library out of this program
#add_library(${PROJECTNAME}_lib ${SOURCES})

add_executable(${PROJECTNAME} src/main.cpp ${SOURCES})
target_link_libraries(${PROJECTNAME}
    ${CMAKE_THREAD_LIBS_INIT}
    ${CMAKE_SOURCE_DIR}/lib/libCPM_lib.so
    ${OpenCV_LIBS}
    )
