/*
 * Minimization.h
 *
 *      Author: Sebastian Lutz
 *  University: Trinity College Dublin
 *      School: Computer Science and Statistics
 *     Project: V-SENSE
 */

#ifndef MINIMIZATION_H_
#define MINIMIZATION_H_

#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <numeric>
#include "constants.h"

typedef double (* vFunctionCall)(std::vector<double> v, void* params); // pointer to function returning double
typedef std::vector<double> (* vFunctionCall2)(std::vector<double> v, void* params);

//static variables for debugging (which is reached: maxIterLineSeach, cg_max_iter or isMin)
extern int reach_ls_iter;
extern int total_line_search;
extern int reach_cg_iter;
extern int reach_isMin_iter;

double energy(std::vector<double> v, std::vector<cv::Vec3d> &means, std::vector<cv::Matx33d> &covs, bool sparse);
double min_f(std::vector<double> &v, void *params);
std::vector<double> min_df(std::vector<double> &v, void* params);
double min_refine_f(std::vector<double> &v, void *params);
std::vector<double> min_refine_df(std::vector<double> &v, void* params);
cv::Vec4d g(std::vector<double> &v, int n, cv::Vec3d color);
std::vector<double> minimizeCG(std::vector<double> x_0, vFunctionCall f, vFunctionCall2 df,
    std::vector<cv::Vec3d> &means, std::vector<cv::Matx33d> &covs, cv::Vec3d color);
std::vector<double> minimizeFCG(std::vector<double> x_0, vFunctionCall f, vFunctionCall2 df,
    std::vector<cv::Vec3d> &means, std::vector<cv::Matx33d> &covs, cv::Vec3d color,  double p, cv::Vec4d lambda, std::vector<double> gt_alpha);
std::vector<double> minimizeMofM(std::vector<double> x_0, vFunctionCall f, vFunctionCall2 df,
    std::vector<cv::Vec3d> &means, std::vector<cv::Matx33d> &covs, cv::Vec3d color, std::vector<double> gt_alpha);

void print_v(std::vector<double> v);

#endif // MINIMIZATION_H_
