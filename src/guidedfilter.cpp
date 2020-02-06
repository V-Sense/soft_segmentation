#include "guidedfilter.h"

static cv::Mat boxfilter(const cv::Mat &I, int r)
{
    cv::Mat result;
    cv::blur(I, result, cv::Size(r, r));
    return result;
}

static cv::Mat convertTo(const cv::Mat &mat, int depth)
{
    if (mat.depth() == depth)
        return mat;

    cv::Mat result;
    mat.convertTo(result, depth);
    return result;
}

class GuidedFilterImpl
{
public:
    virtual ~GuidedFilterImpl() {}

    cv::Mat filter(const cv::Mat &p, int depth);

protected:
    int Idepth;

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const = 0;
};

class GuidedFilterMono : public GuidedFilterImpl
{
public:
    GuidedFilterMono(const cv::Mat &I, int r, double eps);

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:
    int r;
    double eps;
    cv::Mat I, mean_I, var_I;
};

class GuidedFilterColor : public GuidedFilterImpl
{
public:
    GuidedFilterColor(const cv::Mat &I, int r, double eps);

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:
    std::vector<cv::Mat> Ichannels;
    int r;
    double eps;
    cv::Mat mean_I_r, mean_I_g, mean_I_b;
    cv::Mat invrr, invrg, invrb, invgg, invgb, invbb;
};


cv::Mat GuidedFilterImpl::filter(const cv::Mat &p, int depth)
{
    cv::Mat p2 = convertTo(p, Idepth);

    cv::Mat result;
    if (p.channels() == 1)
    {
        result = filterSingleChannel(p2);
    }
    else
    {
        std::vector<cv::Mat> pc;
        cv::split(p2, pc);

        for (std::size_t i = 0; i < pc.size(); ++i)
            pc[i] = filterSingleChannel(pc[i]);

        cv::merge(pc, result);
    }

    return convertTo(result, depth == -1 ? p.depth() : depth);
}

GuidedFilterMono::GuidedFilterMono(const cv::Mat &origI, int r, double eps) : r(r), eps(eps)
{
    if (origI.depth() == CV_32F || origI.depth() == CV_64F)
        I = origI.clone();
    else
        I = convertTo(origI, CV_32F);

    Idepth = I.depth();

    mean_I = boxfilter(I, r);
    cv::Mat mean_II = boxfilter(I.mul(I), r);
    var_I = mean_II - mean_I.mul(mean_I);
}

cv::Mat GuidedFilterMono::filterSingleChannel(const cv::Mat &p) const
{
    cv::Mat mean_p = boxfilter(p, r);
    cv::Mat mean_Ip = boxfilter(I.mul(p), r);
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p); // this is the covariance of (I, p) in each local patch.

    cv::Mat a = cov_Ip / (var_I + eps); // Eqn. (5) in the paper;
    cv::Mat b = mean_p - a.mul(mean_I); // Eqn. (6) in the paper;

    cv::Mat mean_a = boxfilter(a, r);
    cv::Mat mean_b = boxfilter(b, r);

    return mean_a.mul(I) + mean_b;
}

GuidedFilterColor::GuidedFilterColor(const cv::Mat &origI, int r, double eps) : r(r), eps(eps)
{
    cv::Mat I;
    if (origI.depth() == CV_32F || origI.depth() == CV_64F)
        I = origI.clone();
    else
        I = convertTo(origI, CV_32F);

    Idepth = I.depth();

    cv::split(I, Ichannels);

    mean_I_r = boxfilter(Ichannels[0], r);
    mean_I_g = boxfilter(Ichannels[1], r);
    mean_I_b = boxfilter(Ichannels[2], r);

    // variance of I in each local patch: the matrix Sigma in Eqn (14).
    // Note the variance in each local patch is a 3x3 symmetric matrix:
    //           rr, rg, rb
    //   Sigma = rg, gg, gb
    //           rb, gb, bb
    cv::Mat var_I_rr = boxfilter(Ichannels[0].mul(Ichannels[0]), r) - mean_I_r.mul(mean_I_r) + eps;
    cv::Mat var_I_rg = boxfilter(Ichannels[0].mul(Ichannels[1]), r) - mean_I_r.mul(mean_I_g);
    cv::Mat var_I_rb = boxfilter(Ichannels[0].mul(Ichannels[2]), r) - mean_I_r.mul(mean_I_b);
    cv::Mat var_I_gg = boxfilter(Ichannels[1].mul(Ichannels[1]), r) - mean_I_g.mul(mean_I_g) + eps;
    cv::Mat var_I_gb = boxfilter(Ichannels[1].mul(Ichannels[2]), r) - mean_I_g.mul(mean_I_b);
    cv::Mat var_I_bb = boxfilter(Ichannels[2].mul(Ichannels[2]), r) - mean_I_b.mul(mean_I_b) + eps;

    // Inverse of Sigma + eps * I
    invrr = var_I_gg.mul(var_I_bb) - var_I_gb.mul(var_I_gb);
    invrg = var_I_gb.mul(var_I_rb) - var_I_rg.mul(var_I_bb);
    invrb = var_I_rg.mul(var_I_gb) - var_I_gg.mul(var_I_rb);
    invgg = var_I_rr.mul(var_I_bb) - var_I_rb.mul(var_I_rb);
    invgb = var_I_rb.mul(var_I_rg) - var_I_rr.mul(var_I_gb);
    invbb = var_I_rr.mul(var_I_gg) - var_I_rg.mul(var_I_rg);

    cv::Mat covDet = invrr.mul(var_I_rr) + invrg.mul(var_I_rg) + invrb.mul(var_I_rb);

    invrr /= covDet;
    invrg /= covDet;
    invrb /= covDet;
    invgg /= covDet;
    invgb /= covDet;
    invbb /= covDet;
}

cv::Mat GuidedFilterColor::filterSingleChannel(const cv::Mat &p) const
{
    cv::Mat mean_p = boxfilter(p, r);

    cv::Mat mean_Ip_r = boxfilter(Ichannels[0].mul(p), r);
    cv::Mat mean_Ip_g = boxfilter(Ichannels[1].mul(p), r);
    cv::Mat mean_Ip_b = boxfilter(Ichannels[2].mul(p), r);

    // covariance of (I, p) in each local patch.
    cv::Mat cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
    cv::Mat cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
    cv::Mat cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);

    cv::Mat a_r = invrr.mul(cov_Ip_r) + invrg.mul(cov_Ip_g) + invrb.mul(cov_Ip_b);
    cv::Mat a_g = invrg.mul(cov_Ip_r) + invgg.mul(cov_Ip_g) + invgb.mul(cov_Ip_b);
    cv::Mat a_b = invrb.mul(cov_Ip_r) + invgb.mul(cov_Ip_g) + invbb.mul(cov_Ip_b);

    cv::Mat b = mean_p - a_r.mul(mean_I_r) - a_g.mul(mean_I_g) - a_b.mul(mean_I_b); // Eqn. (15) in the paper;

    return (boxfilter(a_r, r).mul(Ichannels[0])
          + boxfilter(a_g, r).mul(Ichannels[1])
          + boxfilter(a_b, r).mul(Ichannels[2])
          + boxfilter(b, r));  // Eqn. (16) in the paper;
}


GuidedFilter::GuidedFilter(const cv::Mat &I, int r, double eps)
{
    CV_Assert(I.channels() == 1 || I.channels() == 3);

    if (I.channels() == 1)
        impl_ = new GuidedFilterMono(I, 2 * r + 1, eps);
    else
        impl_ = new GuidedFilterColor(I, 2 * r + 1, eps);
}

GuidedFilter::~GuidedFilter()
{
    delete impl_;
}

cv::Mat GuidedFilter::filter(const cv::Mat &p, int depth) const
{
    return impl_->filter(p, depth);
}

cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps, int depth)
{
    return GuidedFilter(I, r, eps).filter(p, depth);
}

/**
 * Do the matte regularisation step from Aksoy et al. paper, section 3
 * "Unmixing-Based Soft Color Segmentation for Image Manipulation" [2016]
 * 1. Apply the guided filter to the current given a frame set of layers
 *    following the guide image guide_img, (use current frame as guide image)
 * 2. Regularise resulting alpha values so that they add up to one.
 * Return the regularised layers
 * */
std::vector<cv::Mat> MatteRegularisation(int radius, cv::Mat guide_img, std::vector<cv::Mat> layers)
{
    int n = layers.size(); //number of colours in CM
    std::vector<cv::Mat> filtered_layers; //filtered using guided filter
    std::vector<cv::Mat> regularised_layers; //regularised using matte regularisation

    cv::Mat sum_filtered_alphas(guide_img.rows, guide_img.cols, CV_32FC1, cvScalar(0.0));
    int r = radius;
    double eps = .001; 
    eps *= 255 * 255;
    for(size_t i = 0; i < n; i++){
        //initialise
        cv::Mat layer = layers.at(i);
        cv::Mat layer_split[4];
        cv::Mat ALayer8U, fLayer, guide_img8U;
        cv::Mat ALayer(guide_img.size(),CV_32FC1); //for alpha layer
        cv::Mat fLayer8U(guide_img.size(),CV_8UC1); 

        cv::split(layer, layer_split); //split RGBA channels in layers
        layer_split[3].convertTo(ALayer, CV_32FC1); //convert to float 
        ALayer.convertTo(ALayer8U, CV_8UC1);//convert to uint
        guide_img.convertTo(guide_img8U, CV_8UC3); //convert imput image to uint
        cv::ximgproc::guidedFilter(guide_img8U,ALayer8U,fLayer8U,r,eps,-1); //filter uint alpha layer with uint input image (filtering uint to prevent negative alpha values/decimals)
        fLayer8U.convertTo(fLayer, CV_32FC1);//convert filtered layer back to float so we can manipulate values so they add to 1 (or 255).
        cv::addWeighted(sum_filtered_alphas,1.0,fLayer,1.0/255.0,0,sum_filtered_alphas); //sum all of the filtered alpha layers - this will be used to normalise 
        filtered_layers.push_back(fLayer);
    }	

    // Regularize alpha values so that they add up to one
    for(size_t i = 0; i < n; i++){
        cv::Mat orig_layer = layers.at(i); //these is the original 4D layers - with colour and alpha info
        cv::Mat fillayer = filtered_layers.at(i); // these layers are the new filtered 1D alpha layers
        cv::Mat reg_alpha(guide_img.rows, guide_img.cols, CV_32FC1, cvScalar(0.0));
        //regularise the alpha values so they all add to 1 (or 255, depending on scaling)
        for(size_t i = 0; i < guide_img.rows; i++){
            for(size_t j = 0; j < guide_img.cols; j++){
                reg_alpha.at<float>(i,j) = (1.0/sum_filtered_alphas.at<float>(i,j))*(fillayer.at<float>(i,j));
            }
        }
        
        //transfer these regularized alpha layers to our full colour+alpha layers
        cv::Mat bgra[4];
        cv::Mat bgra2[4];
        cv::Mat reg_layer(guide_img.rows, guide_img.cols, CV_32FC4);
        cv::split(reg_layer,bgra);
        cv::split(orig_layer,bgra2); //split original layers so we can access colour layers
        
        //create new layers with colour and filtered regularised alphas
        bgra2[0].convertTo(bgra[0],bgra[0].type()); //add b to bgra
        bgra2[1].convertTo(bgra[1],bgra[0].type());//add g to bgra
        bgra2[2].convertTo(bgra[2],bgra[0].type());//add r to bgra
        reg_alpha.convertTo(bgra[3],bgra[0].type()); //add a to bgra
        cv::merge(bgra,4,reg_layer);

        reg_layer.convertTo(reg_layer, CV_8UC4);
        regularised_layers.push_back(reg_layer);
    }

    return regularised_layers;
}