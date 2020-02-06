#include "Pixel.h"

Pixel::Pixel(cv::Vec3d color, cv::Point coord){
	this->color = color;
	this->coord = coord;
}

// Get the index of the mean that represents the pixel the closest
int Pixel::minIndex(cv::Vec3d color, std::vector<cv::Vec3d> &means, std::vector<cv::Matx33d> &covs){
	std::vector<double> distances;
	double dist;
	int n = means.size();
	for(size_t i = 0; i < n; i++){
		dist = pow(cv::Mahalanobis(color, means.at(i), (covs.at(i)).inv()),2);
		distances.push_back(dist);
	}
	return min_element(distances.begin(), distances.end()) - distances.begin();
}

/** Unmix the color of the pixel based on the means and covs
 *  means: vector of Vec3d of mean values for the colors in the model
 *  covs: covs of the Matx33d of the inverse covariances of the colors in the model
 * 	x_init layers initialisation thanks to layer n-1 and optical flow, null if first frame
 */
Unmixing Pixel::unmix(std::vector<cv::Vec3d> &means, std::vector<cv::Matx33d> &covs,
	std::vector<double> x_init)
{
	int n = means.size(); //number of layers
	std::vector<double> x_min(4*n);


	int mI = minIndex(this->color, means, covs); //idx of mean that represents the pixel the closest
	std::vector<double> x(4*n); //create vector of zeros of size 4*n
	// Create initial x-vector for minimization
	for(size_t i = 0; i < n; i++){
		x.at(i) = 0;
		x.at(3*i+n) =  means.at(i).val[0];
		x.at(3*i+n+1) = means.at(i).val[1];
		x.at(3*i+n+2) = means.at(i).val[2];
	}

	//intialise layer most representing the pixel to opaque and to pixel colour
	x.at(mI) = 1;
	x.at(3*mI+n) = this->color.val[0];
	x.at(3*mI+n+1) = this->color.val[1];
	x.at(3*mI+n+2) = this->color.val[2];
	
	x_min = minimizeMofM(x, (vFunctionCall)min_f, (vFunctionCall2)min_df, means, covs, this->color, std::vector<double> {0} );

	// Format results
	std::vector<double> alphas;
	std::vector<cv::Vec3d> colors;
	double u1, u2, u3;
	for(size_t i = 0; i < n; i++){
		alphas.push_back(x_min.at(i));
		u1 = x_min.at(3*i+n);
		u2 = x_min.at(3*i+n+1);
		u3 = x_min.at(3*i+n+2);
		colors.push_back(cv::Vec3d(u1,u2,u3));
	}

	Unmixing res;
	res.alphas = alphas;
	res.colors = colors;
	res.coords = this->coord;


	return res;
}


/***********************************************
 * This is the final color refinement step, with the new alpha constraint (Equation 6 of unmixing paper)
 * ********************************************/
Unmixing Pixel::refine(std::vector<cv::Vec3d> &means, std::vector<cv::Matx33d> &covs,
	std::vector<double> x_init)
{
	int n = means.size(); //number of layers
	std::vector<double> x_min(4*n);

	std::vector<double> gt_alphas(n);// save ground truth alphas to variable
	for(int i = 0; i < n; i++){
		gt_alphas.at(i) = x_init.at(i);
	}

		
	x_min = minimizeMofM(x_init, (vFunctionCall)min_refine_f, (vFunctionCall2)min_refine_df, means, covs,this->color, gt_alphas);

	// Format results
	std::vector<double> alphas;
	std::vector<cv::Vec3d> colors;
	double u1, u2, u3;
	for(size_t i = 0; i < n; i++){
		alphas.push_back(x_min.at(i));
		u1 = x_min.at(3*i+n);
		u2 = x_min.at(3*i+n+1);
		u3 = x_min.at(3*i+n+2);
		colors.push_back(cv::Vec3d(u1,u2,u3));
	}

	Unmixing res;
	res.alphas = alphas;
	res.colors = colors;
	res.coords = this->coord;

	return res;
}

