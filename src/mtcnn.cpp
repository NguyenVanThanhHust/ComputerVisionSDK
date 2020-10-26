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

void MTCNN::generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox>& boundingBox_, float scale){
    const int stride = 2;
    const int cellsize = 12;
    //score p
    float *p = score.channel(1);//score.data + score.cstep;
    //float *plocal = location.data;
    Bbox bbox;
    float inv_scale = 1.0f/scale;
    for(int row=0;row<score.h;row++){
        for(int col=0;col<score.w;col++){
            if(*p>threshold[0]){
                bbox.score = *p;
                bbox.x1 = round((stride*col+1)*inv_scale);
                bbox.y1 = round((stride*row+1)*inv_scale);
                bbox.x2 = round((stride*col+1+cellsize)*inv_scale);
                bbox.y2 = round((stride*row+1+cellsize)*inv_scale);
                bbox.area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1);
                const int index = row * score.w + col;
                for(int channel=0;channel<4;channel++){
                    bbox.regreCoord[channel]=location.channel(channel)[index];
                }
                boundingBox_.push_back(bbox);
            }
            p++;
            //plocal++;
        }
    }
}

void MTCNN::nmsTwoBoxs(vector<Bbox>& boundingBox_, vector<Bbox>& previousBox_, const float overlap_threshold, string modelname)
{
	if (boundingBox_.empty()) {
		return;
	}
	sort(boundingBox_.begin(), boundingBox_.end(), cmpScore);
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	//std::cout << boundingBox_.size() << " ";
	for (std::vector<Bbox>::iterator ity = previousBox_.begin(); ity != previousBox_.end(); ity++) {
		for (std::vector<Bbox>::iterator itx = boundingBox_.begin(); itx != boundingBox_.end();) {
			int i = itx - boundingBox_.begin();
			int j = ity - previousBox_.begin();
			maxX = std::max(boundingBox_.at(i).x1, previousBox_.at(j).x1);
			maxY = std::max(boundingBox_.at(i).y1, previousBox_.at(j).y1);
			minX = std::min(boundingBox_.at(i).x2, previousBox_.at(j).x2);
			minY = std::min(boundingBox_.at(i).y2, previousBox_.at(j).y2);
			//maxX1 and maxY1 reuse
			maxX = ((minX - maxX + 1)>0) ? (minX - maxX + 1) : 0;
			maxY = ((minY - maxY + 1)>0) ? (minY - maxY + 1) : 0;
			//IOU reuse for the area of two bbox
			IOU = maxX * maxY;
			if (!modelname.compare("Union"))
				IOU = IOU / (boundingBox_.at(i).area + previousBox_.at(j).area - IOU);
			else if (!modelname.compare("Min")) {
				IOU = IOU / ((boundingBox_.at(i).area < previousBox_.at(j).area) ? boundingBox_.at(i).area : previousBox_.at(j).area);
			}
			if (IOU > overlap_threshold&&boundingBox_.at(i).score>previousBox_.at(j).score) {
			//if (IOU > overlap_threshold) {
				itx = boundingBox_.erase(itx);
			}
			else {
				itx++;
			}
		}
	}
	//std::cout << boundingBox_.size() << std::endl;
}
