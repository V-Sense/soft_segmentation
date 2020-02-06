/*
 * Pixel.h
 *
 *      Author: Sebastian Lutz
 *  University: Trinity College Dublin
 *      School: Computer Science and Statistics
 *     Project: V-SENSE
 */

#ifndef PIXEL_H_
#define PIXEL_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include "Unmixing.h"
#include "Minimization.h"

class Pixel
{
public:
	Pixel(cv::Vec3d color, cv::Point coord);
	Unmixing unmix(std::vector<cv::Vec3d> &means, std::vector<cv::Matx33d> &covs, std::vector<double> x_init);
	Unmixing refine(std::vector<cv::Vec3d> &means, std::vector<cv::Matx33d> &covs, std::vector<double> x_init);
private:
	int minIndex(cv::Vec3d color, std::vector<cv::Vec3d> &means, std::vector<cv::Matx33d> &covs);
	cv::Vec3d color;
	cv::Point coord;
};


#endif // PIXEL_H_
