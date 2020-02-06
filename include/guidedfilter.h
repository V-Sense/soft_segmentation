/*
 * guidedfilter.h
 *
 *  Created on: 9 Mar 2017
 *      Author: https://github.com/atilimcetin/guided-filter
 */

#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>

class GuidedFilterImpl;

class GuidedFilter
{
public:
    GuidedFilter(const cv::Mat &I, int r, double eps);
    ~GuidedFilter();

    cv::Mat filter(const cv::Mat &p, int depth = -1) const;

private:
    GuidedFilterImpl *impl_;
};

cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps, int depth = -1);
std::vector<cv::Mat> MatteRegularisation(int radius, cv::Mat frame, std::vector<cv::Mat> layers);

#endif // GUIDED_FILTER_H
