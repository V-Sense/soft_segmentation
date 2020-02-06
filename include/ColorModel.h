/*
 * ColorModel.h
 *
 *      Author: Sebastian Lutz & Mairéad Grogan & Johanna Barbier
 *  University: Trinity College Dublin
 *      School: Computer Science and Statistics
 *     Project: V-SENSE
 */

#ifndef COLORMODEL_H_
#define COLORMODEL_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include "constants.h"
#include <tuple>
#include "guidedfilter.h"
#include "Pixel.h"
#include "Minimization.h"

void getGlobalColorModel(cv::Mat &image, std::vector<cv::Vec3d> &means, std::vector<cv::Matx33d> &covs, double tau);
bool getGlobalColorModelVideo(std::vector<cv::Vec3d> &means,
	std::vector<cv::Matx33d> &covs, std::vector<cv::Mat> frames, int frames_to_stack, double tau);
	
void saveColorModelAsImage(const char * filename, std::vector<cv::Vec3d> means, std::vector<cv::Matx33d> covs);

float distHistogram(cv::Mat img1, cv::Mat img2);
bool isCMRepresentative(std::vector<cv::Vec3d> means, std::vector<cv::Matx33d> covs, cv::Mat frame);


#endif // COLORMODEL_H_
