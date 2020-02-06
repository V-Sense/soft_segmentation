/*
 * Unmixing.h
 *
 *  Created on: 9 Mar 2017
 *      Author: Sebastian Lutz
 *  University: Trinity College Dublin
 *      School: Computer Science and Statistics
 *     Project: V-SENSE
 */

#ifndef UNMIXING_H_
#define UNMIXING_H_

#include <vector>

struct Unmixing
{
	std::vector<double> alphas;
	std::vector<cv::Vec3d> colors;
	cv::Point coords;
	Unmixing() : alphas(0), colors(0), coords(0,0) {}
};

#endif // UNMIXING_H_
