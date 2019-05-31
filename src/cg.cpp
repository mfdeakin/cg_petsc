
#include <vector>

#include "SyPDSolver.hpp"

int main() {
  PetscInitializeNoArguments();

  SyPDSolver solver_1(3);

  solver_1(
      std::vector<int>{{0, 1, 2}}, std::vector<int>{{0, 1, 2}},
      std::vector<double>{{1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 1.0}});

	// SyPDSolver solver_2 = solver_1;

  std::vector<double> result =
      solver_1.solve(std::vector<double>{{5.0, 2.0, 1.0}});
  for (double v : result) {
    printf("% .6f, ", v);
  }
  printf("\n");

  return 0;
}

// [[1, -1,  1]  [ 3.5]    [ 5 ]
//  [1,  1, -1]  [-0.25] = [ 2 ]
//  [0,  1,  1]] [ 1.25]   [ 1 ]
