# LHC/src/CMakeLists.txt
add_executable(test_1p2l test_1p2l.cpp)
add_executable(test_1p2l_coplanar test_1p2l_coplanar.cpp)
add_executable(test_2p1l test_2p1l.cpp)
add_executable(test_2p1l_coplanar test_2p1l_coplanar.cpp)

# python bindings
add_subdirectory(pybind11)

find_package (Python COMPONENTS Interpreter Development)
message(STATUS "Python_EXECUTABLE: " ${Python_EXECUTABLE})

pybind11_add_module(pybind_ransac MODULE pybind_ransac.cpp)
target_link_libraries(pybind_ransac PUBLIC Eigen3::Eigen)
target_link_libraries(pybind_ransac PUBLIC ceres)
target_compile_features(pybind_ransac PRIVATE cxx_std_17)

