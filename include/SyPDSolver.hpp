
#ifndef SyPDSolver_h
#define SyPDSolver_h

#include <functional>
#include <memory>
#include <vector>

#include <petscksp.h>

// Uses PETSc's conjugate gradient method to solve a symmetric positive definite
// system of equations
class SyPDSolver {
public:
  SyPDSolver(unsigned int n, int max_nonzero = -1);

  double operator()(int i, int j) const;
  std::vector<double> operator()(std::vector<int> rows,
                                 std::vector<int> cols) const;

  double operator()(int i, int j, double val);
  void operator()(const std::vector<int> &rows, const std::vector<int> &cols,
                  const std::vector<double> &vals);

  std::vector<double> solve(std::vector<double> dependent_vars);

private:
  // We need to use PETSc's internal types to get this to work
  // std::unique_ptr<_p_KSP, std::function<void (KSP)>> solver;
  // std::unique_ptr<_p_Mat, std::function<void (Mat)>> system;
  // std::unique_ptr<_p_Vec, std::function<void (Vec)>> dependent, independent;
  std::shared_ptr<_p_KSP> solver;
  std::shared_ptr<_p_Mat> system;
  std::shared_ptr<_p_Vec> dependent, independent;
  unsigned int num_eqns;
};

#endif // SyPDSolver_h
