#include "mtcnn.h"

#include "models/det1_bin.h"
#include "models/det2_bin.h"
#include "models/det3_bin.h"

#include "models/det1_param.h"
#include "models/det2_param.h"
#include "models/det3_param.h"

using namespace mtcnn;
using namespace std;

bool cmpScore(Bbox first_bbox, Bbox second_bbox)
{
    if(first_bbox.score < second_bbox.score)
    {
        return true;
    }
    return false;
}

bool cmpArea(Bbox first_bbox, Bbox second_bbox)
{
    if(first_bbox.area < second_bbox.area)
    {
        return true;
    }
    return false;
}

// constructor function
MTCNN::MTCNN()
{
    // Load the models from memory
    Pnet.load_param_mem((char *) det1_param);
    Pnet.load_model(det1_bin);

    Rnet.load_param_mem((char *) det2_param);
    Rnet.load_model(det2_bin);

    Onet.load_param_mem((char *) det3_param);
    Onet.load_model(det3_bin);
}

MTCNN::MTCNN(const string &model_path) {

	std::vector<std::string> param_files = {
		model_path+"/det1.param",
		model_path+"/det2.param",
		model_path+"/det3.param"
	};

	std::vector<std::string> bin_files = {
		model_path+"/det1.bin",
		model_path+"/det2.bin",
		model_path+"/det3.bin"
	};

	Pnet.load_param(param_files[0].data());
	Pnet.load_model(bin_files[0].data());
	Rnet.load_param(param_files[1].data());
	Rnet.load_model(bin_files[1].data());
	Onet.load_param(param_files[2].data());
	Onet.load_model(bin_files[2].data());
}

MTCNN::MTCNN(const std::vector<std::string> param_files, const std::vector<std::string> bin_files){
    Pnet.load_param(param_files[0].data());
    Pnet.load_model(bin_files[0].data());
    Rnet.load_param(param_files[1].data());
    Rnet.load_model(bin_files[1].data());
    Onet.load_param(param_files[2].data());
    Onet.load_model(bin_files[2].data());
}

MTCNN::~MTCNN(){
    Pnet.clear();
    Rnet.clear();
    Onet.clear();
}

void MTCNN::SetMinFace(int minSize){
	minsize = minSize;
}
