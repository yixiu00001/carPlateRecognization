#ifndef EASYPR_CORE_CAFFERECOGNISE_H_
#define EASYPR_CORE_CAFFERECOGNISE_H_

#include "easypr/util/kv.h"
#include <memory>
#include "opencv2/opencv.hpp"
#include <opencv2/dnn.hpp>
#include <fstream>
#include <sstream>
#include <vector>

#include "easypr/core/plate.hpp"
#include "easypr/config.h"
#include "easypr/core/plate_detect.h"
#include "easypr/core/chars_recognise.h"
#include "easypr/core/plate_judge.h"
#include "easypr/core/chars_segment.h"
#include "easypr/core/chars_identify.h"
#include "easypr/core/plate_recognize.h"
#include "easypr/core/plate_locate.h"

using namespace std;
using namespace cv;
using namespace easypr;

namespace caffepr
{
	class CaffeRecognise : public CPlateDetect, public CCharsRecognise
	{
	public:
		CaffeRecognise();
		CaffeRecognise(string);
		~CaffeRecognise();
		void LoadModel(const char* modelTxt, const char*modelBin);
		void LoadChineseModel(const char* modelTxt, const char*modelBin);

		std::pair<std::string, std::string> identify(cv::Mat input, bool isChinese = false);

		Mat doResizeImg40(Mat& img);
		Mat doResizeImg28(Mat& img);
		void getMaxClass(dnn::Blob &probBlob, int *classId, double *classProb);
		std::vector<String> readClassNames(const char* filename);
		string recongnisePlate(Mat &img);
		string recongnisePlateChinese(Mat &img);
		int recogniseCaffe(cv::Mat plate, std::string& plateLicense);
		int plateRecognizeMain(Mat src, std::vector<CPlate> &plateVecOut);
		//string plateRecongnizeCNN(char* imagebuffer, int size);
		string process(char*, int);
		inline void setResultShow(bool param) { m_showResult = param; }
		inline bool getResultShow() const { return m_showResult; }
	private:
		String modelTxt;
		String modelBin;
		const char* filename;
		bool m_showResult;

		std::shared_ptr<easypr::Kv> kv_;
		Ptr<dnn::Importer> importer;
		Ptr<dnn::Importer> importerChinese;

		easypr::CCharsSegment *m_charsSegment;
	public:
		dnn::Net net_;
		dnn::Net netChinese_;

	};
}

#endif