
#include "SyPDSolver.hpp"

#include <iostream>

#include <assert.h>

#include <petscmat.h>
#include <petscvec.h>

KSP allocate_sym_pd_solver() {
  KSP solver;
  KSPCreate(MPI_COMM_SELF, &solver);
  KSPSetType(solver, KSPCG);
  KSPSetInitialGuessNonzero(solver, PETSC_FALSE);
  return solver;
}

Mat allocate_sym_pd_mat(unsigned int n, int max_nonzero) {
  if (max_nonzero == -1) {
    max_nonzero = n;
  }
  Mat system;
  MatCreateSeqAIJ(MPI_COMM_SELF, n, n, max_nonzero, NULL, &system);
  return system;
}

Vec allocate_vec(unsigned int n) {
  Vec v;
  VecCreateSeq(MPI_COMM_SELF, n, &v);
  return v;
}

SyPDSolver::SyPDSolver(unsigned int n, int max_nonzero)
    : solver(allocate_sym_pd_solver(),
             std::function<void(KSP)>([](KSP s) { KSPDestroy(&s); })),
      system(allocate_sym_pd_mat(n, max_nonzero),
             std::function<void(Mat)>([](Mat s) { MatDestroy(&s); })),
      dependent(allocate_vec(n),
                std::function<void(Vec)>([](Vec s) { VecDestroy(&s); })),
      independent(allocate_vec(n),
                  std::function<void(Vec)>([](Vec s) { VecDestroy(&s); })),
      num_eqns(n) {}

double SyPDSolver::operator()(int i, int j) const {
  double val;
  MatGetValues(system.get(), 1, &i, 1, &j, &val);
  return val;
}

std::vector<double> SyPDSolver::operator()(std::vector<int> rows,
                                           std::vector<int> cols) const {
  assert(rows.size() <= num_eqns);
  assert(cols.size() <= num_eqns);
  std::vector<double> vals(rows.size() * cols.size());
  MatGetValues(system.get(), rows.size(), rows.data(), cols.size(), cols.data(),
               vals.data());
  return vals;
}

double SyPDSolver::operator()(int i, int j, double val) {
  assert(i >= 0);
  assert(i < num_eqns);
  assert(j >= 0);
  assert(j < num_eqns);
  MatSetValues(system.get(), 1, &i, 1, &j, &val, INSERT_VALUES);
  return val;
}

void SyPDSolver::operator()(const std::vector<int> &rows,
                            const std::vector<int> &cols,
                            const std::vector<double> &vals) {
  assert(vals.size() == rows.size() * cols.size());
  MatSetValues(system.get(), rows.size(), rows.data(), cols.size(), cols.data(),
               vals.data(), INSERT_VALUES);
}

std::vector<double> SyPDSolver::solve(std::vector<double> dependent_vars) {
  assert(dependent_vars.size() == num_eqns);
  std::vector<int> indices(num_eqns);
  for (int i = 0; i < num_eqns; i++) {
    indices[i] = i;
  }
  VecSetValues(dependent.get(), num_eqns, indices.data(), dependent_vars.data(),
               INSERT_VALUES);

  VecAssemblyBegin(dependent.get());
  VecAssemblyEnd(dependent.get());

  MatAssemblyBegin(system.get(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(system.get(), MAT_FINAL_ASSEMBLY);

  KSPSetOperators(solver.get(), system.get(), system.get());
  KSPSetFromOptions(solver.get());
  KSPSetUp(solver.get());

  KSPSolve(solver.get(), dependent.get(), independent.get());
  std::vector<double> independent_vars(num_eqns);
  VecGetValues(independent.get(), num_eqns, indices.data(),
               independent_vars.data());
  return independent_vars;
}
