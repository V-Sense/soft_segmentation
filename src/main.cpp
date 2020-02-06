#include <opencv2/opencv.hpp>
#include "Pixel.h"
#include "Unmixing.h"
#include "ThreadPool.h"
#include <chrono>
#include <vector>
#include <string>
#include "guidedfilter.h"
#include "ColorModel.h"
#include <sys/stat.h> 


int makeDirectories(std::string result_folder_name)
{
    //create folder to save output layers and videos cleanly
    int status;
    status = mkdir(result_folder_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if( status != 0){
		std::cout << "Error mkdir: " <<strerror(errno)<<std::endl;
	    return EXIT_FAILURE;
	}
    result_folder_name += "/";
	status = mkdir((result_folder_name +"output_layers").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if( status != 0){
		std::cout << "Error mkdir: " <<strerror(errno)<<std::endl;
	    return EXIT_FAILURE;
	}
	status = mkdir((result_folder_name +"sum_frames").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if( status != 0){
		std::cout << "Error mkdir: " <<strerror(errno)<<std::endl;
	    return EXIT_FAILURE;
	}
    return EXIT_SUCCESS;
}

void saveLayers(std::string dir, std::vector<cv::Mat> layers, std::string step)
{
    for (int l = 0; l<layers.size(); l++)
        {
            std::string l_str = std::string(2 - std::to_string(l).length(), '0') + std::to_string(l);
            imwrite(dir + "/output_layers/" + step + l_str + ".png",
                layers[l]);
        }
}

void sumLayers(cv::Mat sum, std::vector<cv::Mat> layers)
{
	int n = layers.size();

	//for every pixel
	for(size_t i = 0; i < sum.rows; i++)
	{
		for(size_t j = 0; j < sum.cols; j++)
		{
			double alpha_unit;
			double sum_u1 = 0.0, sum_u2 = 0.0, sum_u3 = 0.0, sum_alpha = 0.0;
			cv::Vec4b& pix = sum.at<cv::Vec4b>(i,j);
			//for every layer
			for(size_t l = 0; l < n; l++){
				cv::Mat layer = layers.at(l);
				cv::Vec4b v = layer.at<cv::Vec4b>(i,j);

				alpha_unit = v[3] / 255.0;
				sum_u1 += alpha_unit*v[0]; //B_i * alpha_i
				sum_u2 += alpha_unit*v[1]; //G_i * alpha_i
				sum_u3 += alpha_unit*v[2]; //R_i * alpha_i
				sum_alpha += v[3];
			}

			// clip because 256 becomes 1 in uchar
			pix[0] = (sum_u1>255)?255:sum_u1;
			pix[1] = (sum_u2>255)?255:sum_u2;
			pix[2] = (sum_u3>255)?255:sum_u3;
			pix[3] = (sum_alpha>255)?255:sum_alpha;
		}
	}
}



int main( int argc, char** argv )
{
	double tau;
	if(argc == 1){
		std::cout << "Usage: ./SoftSegmentation <path/to/image/> <tau_parameterer(optional)> <output_dir>" << std::endl; 
		return -1;
	}else if(argc == 2){
		tau = constants::tau;
	}else {
		tau = std::stod(argv[2]);
	}

    std::vector<cv::Vec3d> means;
    std::vector<cv::Matx33d> covs;

    // Load image
	char* imageName = argv[1];
    char* output_dir_name = argv[3];
	std::string imageName_short(imageName);//mg_added
  	std::string imageName_s(imageName_short.begin()+3, imageName_short.end()-4);
	std::string dir_name(output_dir_name);

	//create directories that we'll save images into
	if(makeDirectories(std::string("../") + dir_name) != 0){
        std::cout <<"Error making directories"<<std::endl;
        return false;
    }

	  
	//Read input image and write it to output directory
	cv::Mat image;
	image = cv::imread(imageName, 1);
	cv::imwrite(std::string("../") + dir_name + "/input_image.png", image); //save input image for reference
	if( !image.data ){
	   printf( " No image data \n " );
	   return -1;
	}


	/*************************************************************
	 *            To Begin: Compute Global Color Model             *
	 *************************************************************/
	//create terminal output
	std::cout << "" << std::endl;
	std::cout << "Colour Model Estimation Step..." << std::endl;
	std::cout << "" << std::endl;


    auto t_start_CM = std::chrono::high_resolution_clock::now();

    means.clear();
    covs.clear();

    std::cout << "Calculating Global Color Model with tau = " << tau<<"."<< std::endl;
            
    //convert video_frames from CV_U8C3 to CV_64FC3 
    cv::Mat img;
	image.convertTo(img,CV_64F,1.0/255.0);

	//compute colour model
    getGlobalColorModel(img, means, covs, tau);

	//output to terminal and save image
    std::cout << "Found " << means.size() << " colors." << std::endl;
    saveColorModelAsImage((std::string("../") + dir_name + "/CM" + ".png").c_str(), means, covs);

	std::cout << "" << std::endl;
	std::cout << "Colour Model Estimation Step: Done. " << std::endl;
    std::cout << "" << std::endl;
	//output mean and covariances for debugging
    //std::cout << "Means:"<<std::endl;
    //for (size_t i=0; i< means.size(); i++){
    //   std::cout << "mean :" << means[i] <<std::endl;
    //    std::cout << "cov : " << covs[i] <<std::endl;
    //}

	///////////////////////////////////////////////// end of step one



	/*************************************************************
	*               Step 1. Sparse Colour Unmixing Step              *
    *************************************************************/

    std::cout << "Step 1/3: Colour Unmixing..." << std::endl;
	std::cout << "" << std::endl;
   	//initialise variables
	int n = means.size(); //nb of layers
	int rows = image.rows;
	int cols = image.cols;
    cv::Vec3d color;
	std::vector<cv::Mat> layers;
	for(size_t i = 0; i < n; i++){
		cv::Mat layer(image.rows, image.cols, CV_8UC4);
		layers.push_back(layer);
	}



    // Create Thread pool
	int num_thread = 8;
    //std::cout << "Using Thread Pool with " << num_thread << " threads." << std::endl;
    ThreadPool pool(num_thread); 
    std::vector<std::future<Unmixing>> results; //where we will store unmixing results


    //for each frame unmixing duration
	//auto t_start_unmix = std::chrono::high_resolution_clock::now(); //which is also t_end_flow

    // Parse image and add one task per pixel to thread pool
    for(size_t i = 0; i < image.rows; i++){
        for(size_t j = 0; j < image.cols; j++){

            color = image.at<cv::Vec3b>(i,j);
            color = color/255;
            Pixel p = Pixel(color, cv::Point(j,i));
                
            std::vector<double> x_init(0); //create vector of zeros of size 0
            results.emplace_back(pool.enqueue( 
            [](Pixel p, std::vector<cv::Vec3d> means, std::vector<cv::Matx33d> covs,std::vector<double> x_init)
            {return p.unmix(means, covs, x_init);}, p, means, covs, x_init));
        }
    }

	// Get results from thread pool and create layers
	std::cout << "Percentage complete:"  << std::endl;
    int num = 0;
    int transp_pix_sup = 0, transp_pix_inf = 0;
    float progress = 0.0;
    int barWidth = 70;
    for(auto && result : results)
    {
        num++;
        //show progress percentage
        progress = float(num)/float(cols*rows);
        std::cout<< int(progress * 100.0) << " %\r";
        std::cout.flush();

        // Create layers from results
        int sum_alpha = 0; //verify alpha constraint
        Unmixing u = result.get(); 
            
        for(size_t i = 0; i < n; i++){
            cv::Mat layer = layers.at(i);
            cv::Vec4b& v = layer.at<cv::Vec4b>(u.coords);

            v[0] = u.colors.at(i).val[0]*255;
            v[1] = u.colors.at(i).val[1]*255;
            v[2] = u.colors.at(i).val[2]*255;
            v[3] = u.alphas.at(i)*255; 
            sum_alpha += floor(u.alphas.at(i)*255);
        }
    }
    std::cout<< "100 %" <<std::endl;
	std::cout << "" << std::endl;
	std::cout<< "Step 1/3: Done." <<std::endl;
	std::cout << "" << std::endl;


    //duration of layer filtering
    auto t_start_filters = std::chrono::high_resolution_clock::now();


    // save images
    saveLayers((std::string("../") + dir_name).c_str(),  layers, "step1_");



	/*************************************************************
    *               Step 2. Matte Regularization Step                *
    *************************************************************/
    std::cout << "Step 2/3: Matte Regularisation..."   << std::endl;
	//Matte regularisation with guided filter
	std::vector<cv::Mat> output_layers;
	int r = ((60.0/(sqrt(1000000.0/(cols*rows))))+1.0); //radius for filter based on one used in unmixing paper
    std::cout << "Filter radius has size: " << r <<std::endl; 

    output_layers = MatteRegularisation(r, image, layers); //filter and regularise the alpha values

    saveLayers((std::string("../") + dir_name).c_str(),  output_layers, "step2_");
	
	std::cout << "" << std::endl;
	std::cout << "Step 2/3: Done."  << std::endl;
	std::cout << "" << std::endl;



	/*************************************************************
    *               3. Colour Refinement Step         	         *
    *************************************************************/
    std::cout << "Step 3/3: Colour Refinement..."  << std::endl;
	std::cout << "" << std::endl;


    //std::cout << "Using Thread Pool with " << num_thread << " threads." << std::endl;
    ThreadPool pool_r(num_thread);
    std::vector<std::future<Unmixing>> refine_results;

    // Parse image and add one task per pixel to thread pool
    for(size_t i = 0; i < rows; i++){
       for(size_t j = 0; j < cols; j++){

            color = image.at<cv::Vec3b>(i,j);
            color = color/255;
            Pixel p = Pixel(color, cv::Point(j,i));
                
            std::vector<double> x_init_r(4*n);
            for(size_t lay = 0; lay < n; lay++){ //for each layer of this frame, initialise colours 
                x_init_r.at(lay) = output_layers[lay].at<cv::Vec4b>(i,j)[3] / 255.0f;
                x_init_r.at(3*lay+n) =  output_layers[lay].at<cv::Vec4b>(i,j)[0] / 255.0f;
                x_init_r.at(3*lay+n+1) = output_layers[lay].at<cv::Vec4b>(i,j)[1] / 255.0f;
                x_init_r.at(3*lay+n+2) = output_layers[lay].at<cv::Vec4b>(i,j)[2] / 255.0f;
            }
            refine_results.emplace_back(pool_r.enqueue( 
            [](Pixel p, std::vector<cv::Vec3d> means, std::vector<cv::Matx33d> covs,std::vector<double> x_init_r)
            {return p.refine(means, covs, x_init_r);}, p, means, covs, x_init_r));
        }
    }
        
    // Get results from thread pool and create layers
    num = 0;
    transp_pix_sup = 0;
	transp_pix_inf = 0;
    progress = 0.0;
    barWidth = 70;
	std::cout << "Percentage complete:"  << std::endl;
    for(auto && result : refine_results)
    {
        num++;
        //show progress percentage
        progress = float(num)/float(cols*rows);
        std::cout<< int(progress * 100.0) << " %\r";
        std::cout.flush();

        // Create layers from results
        int sum_alpha = 0; //verify alpha constraint
        Unmixing u = result.get();
            
        for(size_t i = 0; i < n; i++){
    		cv::Mat layer = layers.at(i);
        	cv::Vec4b& v = layer.at<cv::Vec4b>(u.coords);
            v[0] = u.colors.at(i).val[0]*255;
            v[1] = u.colors.at(i).val[1]*255;
            v[2] = u.colors.at(i).val[2]*255;
            v[3] = u.alphas.at(i)*255; 
            sum_alpha += floor(u.alphas.at(i)*255);
        }
        //std::cout <<   "sum_alpha" << sum_alpha << std::endl;
    }
    std::cout<< "100 %" <<std::endl;
	std::cout << "" << std::endl;
	std::cout<< "Step 3/3: Done. " <<std::endl;
	std::cout << "" << std::endl;
	/*************************************************************
    *                  Save Final Layers          	         *
    *************************************************************/
    //compute sum of layers
	cv::Mat sum_layers(image.rows, image.cols, CV_8UC4, 0.0);
    sumLayers(sum_layers, layers);

    //save all layers as images
    saveLayers((std::string("../") + dir_name).c_str(),  layers, "FinalLayers_" ); 

	//save the sum of the layers as an image
    cv::imwrite((std::string("../") + dir_name).c_str() + std::string("/sum_frames/sum_frame.png"), sum_layers);

	std::cout<< "Final layers saved as images. " <<std::endl;

	/************************************************************/


	return 0;
}
