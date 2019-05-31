
cmake_minimum_required(VERSION 3.14)

add_executable(cg src/cg.cpp src/SyPDSolver.cpp)

target_include_directories(cg PUBLIC include
  /home/michael/programming/petsc/include
  /home/michael/programming/petsc/arch-linux2-c-debug/include)

target_link_directories(cg PUBLIC /home/michael/programming/petsc/arch-linux2-c-debug/lib)
target_link_libraries(cg petsc)
