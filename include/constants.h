/*
 * constants.h
 *
 *      Author: Sebastian Lutz
 *  University: Trinity College Dublin
 *      School: Computer Science and Statistics
 *     Project: V-SENSE
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

namespace constants {
constexpr double beta = 10;
constexpr double gamma = 0.25;
constexpr double sigma = 100; //set to 10 in paper - mg increased to 100
constexpr double eps = 0.0001;
constexpr double step_size = 0.01;
constexpr double tol = 0.000001;
constexpr double ls_max_iter = 10; //line_search max iteration
constexpr double cg_max_iter = 30; //conjugate gradient max iteration
constexpr double isMin_max_iter = 2;
constexpr double ls_tau = 0.25;
constexpr double ls_c = 0.05; //
constexpr double tau = 13; //mg found 13 was best with this method
}


#endif // CONSTANTS_H_
