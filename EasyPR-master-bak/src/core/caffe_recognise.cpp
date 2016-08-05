#include "easypr/core/caffe_recognise.h"
#include "easypr/config.h"
#include "easypr/util/util.h"
#include "direct.h"
using namespace easypr;
namespace caffepr
{

	CaffeRecognise::CaffeRecognise() 
	{
		//以下默认构造函数已被注释
		PlateJudge::instance()->LoadModel(kDefaultSvmPath);

		//以下四个应该没有
		CharsIdentify::instance()->LoadModel(kDefaultAnnPath);
		CharsIdentify::instance()->LoadChineseModel(kChineseAnnPath);
		CharsIdentify::instance()->kv_ = std::shared_ptr<Kv>(new Kv);
		CharsIdentify::instance()->kv_->load("etc/province_mapping");

		kv_ = std::shared_ptr<Kv>(new Kv);
		kv_->load("resources/model/province_mapping");
		m_charsSegment = new CCharsSegment();

		LoadModel(kDefaultCnnModelTxt, kDefaultCnnModelBin);
		//setDetectType(PR_DETECT_CMSER | PR_DETECT_COLOR | PR_DETECT_SOBEL);
		setDetectType(PR_DETECT_CMSER);
		setPDLifemode(true);
		setResultShow(false);
		

	}

	CaffeRecognise::CaffeRecognise(string path)
	{
		//string path = "D:/JAVA/workspaceneo/img-demo/imgprocessing/target/test-classes/model";
		//以下默认构造函数已被注释,不然返回空值无法判断
		PlateJudge::instance()->LoadModel(path+"/svm.xml");
		
		//以下四个我不知道还需要不，这里只能加载。。。
		CharsIdentify::instance()->LoadModel(path+"/ann.xml");
		CharsIdentify::instance()->LoadChineseModel(path+"/ann_chinese.xml");
		CharsIdentify::instance()->kv_ = std::shared_ptr<Kv>(new Kv);
		CharsIdentify::instance()->kv_->load(path + "/province_mapping");


		kv_ = std::shared_ptr<Kv>(new Kv);
		kv_->load(path + "/province_mapping");
		m_charsSegment = new CCharsSegment();

		LoadModel((path+"/character.prototxt").c_str(), (path+"/character.caffemodel").c_str());
		//setDetectType(PR_DETECT_CMSER | PR_DETECT_COLOR | PR_DETECT_SOBEL);
		setDetectType(PR_DETECT_COLOR | PR_DETECT_CMSER);
		setPDLifemode(true);
		setResultShow(false);
	
	}


	CaffeRecognise::~CaffeRecognise()
	{
		SAFE_RELEASE(m_charsSegment);
	}
	void CaffeRecognise::LoadModel(const char* modelTxt, const char*modelBin)
	{
		try                                     //Try to import Caffe GoogleNet model
		{
			importer = dnn::createCaffeImporter(modelTxt, modelBin);
		}
		catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
		{
			std::cerr << "Load model failed!" << endl;
			std::cerr << err.msg << std::endl;
		}
		
		if (!importer)
		{
			std::cerr << "Can't load network by using the following files: " << std::endl;
			std::cerr << "prototxt:   " << modelTxt << std::endl;
			std::cerr << "caffemodel: " << modelBin << std::endl;
			std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
			std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
			exit(-1);
		}

		importer->populateNet(net_);
		importer.release();
	}

	void CaffeRecognise::LoadChineseModel(const char* modelTxt, const char*modelBin)
	{
		try                                     //Try to import Caffe GoogleNet model
		{
			importerChinese = dnn::createCaffeImporter(modelTxt, modelBin);
		}
		catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
		{
			cout << "Load model failed!" << endl;
			std::cerr << err.msg << std::endl;
		}


		if (!importerChinese)
		{
			std::cerr << "Can't load network by using the following files: " << std::endl;
			std::cerr << "prototxt:   " << modelTxt << std::endl;
			std::cerr << "caffemodel: " << modelBin << std::endl;
			std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
			std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
			exit(-1);
		}


		importerChinese->populateNet(netChinese_);
		importerChinese.release();
	}
	Mat CaffeRecognise::doResizeImg40(Mat& img)
	{
		// resize(img,img,Size(32,32));
		//img.copyTo(binary_img);
		int imgc = img.rows;
		Mat in_large = Mat::zeros(Size(imgc, imgc), img.type());//建立自适应黑板

		Mat in_large40 = Mat::zeros(Size(40, 40), img.type());//建立40黑板

		float x1 = in_large.cols / 2 - img.cols / 2;//两个图像的中心点差x坐标
		float y1 = in_large.rows / 2 - img.rows / 2;//两个图像的中心点差y坐标
		//将图像A（20×20）按照上下左右各空出x或y的像素宽，复制到B（28×28）。
		copyMakeBorder(img, in_large, y1, y1, x1, x1, BORDER_CONSTANT, Scalar::all(0));

		resize(in_large, in_large, Size(32, 32));
		float x = in_large40.cols / 2 - in_large.cols / 2;//两个图像的中心点差x坐标
		float y = in_large40.rows / 2 - in_large.rows / 2;//两个图像的中心点差y坐标
		//将图像A（32×32）按照上下左右各空出x或y的像素宽，复制到B（40×40）。
		copyMakeBorder(in_large, in_large40, y, y, x, x, BORDER_CONSTANT, Scalar::all(0));

		resize(in_large40, in_large40, Size(40, 40));//由于有个bug，重新标准化

		return in_large40;
	}
	Mat CaffeRecognise::doResizeImg28(Mat& img)
	{
		// resize(img,img,Size(32,32));
		//img.copyTo(binary_img);
		int imgc = img.rows;
		Mat in_large = Mat::zeros(Size(imgc, imgc), img.type());//建立自适应黑板

		Mat in_large40 = Mat::zeros(Size(28, 28), img.type());//建立40黑板

		float x1 = in_large.cols / 2 - img.cols / 2;//两个图像的中心点差x坐标
		float y1 = in_large.rows / 2 - img.rows / 2;//两个图像的中心点差y坐标
		//将图像A（20×20）按照上下左右各空出x或y的像素宽，复制到B（28×28）。
		copyMakeBorder(img, in_large, y1, y1, x1, x1, BORDER_CONSTANT, Scalar::all(0));

		resize(in_large, in_large, Size(20, 20));
		float x = in_large40.cols / 2 - in_large.cols / 2;//两个图像的中心点差x坐标
		float y = in_large40.rows / 2 - in_large.rows / 2;//两个图像的中心点差y坐标
		//将图像A（32×32）按照上下左右各空出x或y的像素宽，复制到B（40×40）。
		copyMakeBorder(in_large, in_large40, y, y, x, x, BORDER_CONSTANT, Scalar::all(0));

		resize(in_large40, in_large40, Size(28, 28));//由于有个bug，重新标准化

		return in_large40;
	}
	void CaffeRecognise::getMaxClass(dnn::Blob &probBlob, int *classId, double *classProb)
	{
		// Mat probMat = probBlob.matRefConst().reshape(1, 1); //reshape the blob to 1x1000 matrix
		Mat probMat = probBlob.matRefConst().reshape(1, 1);
		Point classNumber;
		minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
		*classId = classNumber.x;
	}


	//const char * caffepr::CaffeRecognise::filename = "heh.txt";
	std::vector<String> CaffeRecognise::readClassNames(const char *filename)
	{
		std::vector<String> classNames;
		std::ifstream fp(filename);
		if (!fp.is_open())
		{
			std::cerr << "File with classes labels not found: " << filename << std::endl;
			exit(-1);
		}
		std::string name;
		while (!fp.eof())
		{
			std::getline(fp, name);
			if (name.length())
				classNames.push_back(name.substr(name.find(' ') + 1));
		}
		fp.close();
		return classNames;
	}

	string CaffeRecognise::recongnisePlate(Mat &img)
	{

		            //We don't need importer anymore
		
		Mat resize_img = doResizeImg28(img);

		if (resize_img.empty())
		{
			std::cerr << "Can't read image" << std::endl;
			exit(-1);
		}
		dnn::Blob inputBlob = dnn::Blob(resize_img);   //Convert Mat to dnn::Blob image batch
		net_.setBlob(".data", inputBlob);        //set the network input

		net_.forward();                          //compute output
		dnn::Blob prob = net_.getBlob("prob");

		int classId;
		double classProb;
		getMaxClass(prob, &classId, &classProb);//find the best class
		//cout << "classID=" << classId << " classProb=" << classProb << endl;
		//this->filename = "heh.txt";
		//std::vector<String> classNames = readClassNames(filename);
		if (classId < kCharactersNumber)
		{
			return kChars[classId];
		}
		else
		{
			//auto index = classId + kCharsTotalNumber - kChineseNumber;
			const char* key = kChars[classId];
			std::string s = key;
			std::string province = kv_->get(s);
			//cout << "chinese =" << key << " " << province << endl;
			return province;
		}
	}
	string CaffeRecognise::recongnisePlateChinese(Mat &img)
	{

		//We don't need importer anymore

		Mat resize_img = doResizeImg28(img);

		if (resize_img.empty())
		{
			std::cerr << "Can't read image" << std::endl;
			exit(-1);
		}
		dnn::Blob inputBlob = dnn::Blob(resize_img);   //Convert Mat to dnn::Blob image batch
		net_.setBlob(".data", inputBlob);        //set the network input

		net_.forward();                          //compute output
		dnn::Blob prob = net_.getBlob("prob");

		int classId;
		double classProb;
		getMaxClass(prob, &classId, &classProb);//find the best class
		//cout << "classID=" << classId << " classProb=" << classProb << endl;
		//this->filename = "heh.txt";
		//std::vector<String> classNames = readClassNames(filename);
		if (classId < kCharactersNumber)
		{
			return "";
		}
		else
		{
			//auto index = classId + kCharsTotalNumber - kChineseNumber;
			const char* key = kChars[classId];
			std::string s = key;
			std::string province = kv_->get(s);
			//cout << "chinese =" << key << " " << province << endl;
			return province;
		}
	}
	int CaffeRecognise::recogniseCaffe(cv::Mat plate, std::string& plateLicense)
	{
		std::vector<Mat> matChars;

		int result = m_charsSegment->charsSegment(plate, matChars);
		
		//std::cout << "charsSegment:" << result << std::endl;

		if (result == 0) {

			int num = matChars.size();
			//cout << "num=" << num << endl;
			for (int j = 0; j < num; j++)
			{
				Mat charMat = matChars.at(j);
				bool isChinses = false;
				float maxVal = 0;
				if (j == 0) 
				{
					bool judge = true;
					isChinses = true;
					auto character = recongnisePlateChinese(charMat);
					//cout << j << "=" << character << endl;
					//auto character = recongnisePlateChinese(charMat);
					if (!character.empty())
						plateLicense.append(character);
					else
						return -1;
				}
				else 
				{
					isChinses = false;
					auto character =recongnisePlate(charMat);
					//cout << j << "=" << character << endl;
					if (!character.empty())
						plateLicense.append(character);
				}
			}

		}
		if (plateLicense.size() < 7) {
			return -1;
		}

		return result;
	}

	int CaffeRecognise::plateRecognizeMain(Mat src, std::vector<CPlate>& plateVecOut)
	{
		int result = 0;

		std::vector<CPlate> plateVec;		
		
		int resultPD = plateDetect(src, plateVec);
		
		if (resultPD == 0)
		{
			size_t num = plateVec.size();
			int index = 0;

			for (size_t j = 0; j < num; j++)
			{
				CPlate item = plateVec.at(j);
				Mat plateMat = item.getPlateMat();

				Color color = item.getPlateColor();
				if (color == UNKNOWN)
				{
					color = getPlateType(plateMat, true);
					item.setPlateColor(color);
				}

				std::string plateColor = getPlateColor(color);
				if (0)
				{
					std::cout << "plateColor:" << plateColor << std::endl;
				}
				
				std::string plateIdentify = "";
				resultPD = recogniseCaffe(item.getPlateMat(), plateIdentify);
				
				if (resultPD == 0)
				{
					std::string license = plateColor + ":" + plateIdentify;
					item.setPlateStr(license);
					plateVecOut.push_back(item);
				}
				//else
				//{
				//	//std::string license = plateColor;
				//	//item.setPlateStr(license);
				//	//plateVecOut.push_back(item);
				//	//if (0)
				//	//{
				//	//	std::cout << "resultCR:" << resultCR << std::endl;
				//	//}
				//}
			}
			setResultShow(false);

			if (getResultShow()) {
				Mat result;
				src.copyTo(result);

				for (size_t j = 0; j < num; j++)
				{
					CPlate item = plateVec[j];
					Mat plateMat = item.getPlateMat();

					int height = 36;
					int width = 136;
					if (height * index + height < result.rows)
					{
						Mat imageRoi = result(Rect(0, 0 + height * index, width, height));
						addWeighted(imageRoi, 0, plateMat, 1, 0, imageRoi);
					}
					index++;

					RotatedRect minRect = item.getPlatePos();
					Point2f rect_points[4];
					minRect.points(rect_points);

					Scalar lineColor = Scalar(255, 255, 255);

					if (item.getPlateLocateType() == SOBEL) lineColor = Scalar(255, 0, 0);
					if (item.getPlateLocateType() == COLOR) lineColor = Scalar(0, 255, 0);
					if (item.getPlateLocateType() == CMSER) lineColor = Scalar(0, 0, 255);

					for (int j = 0; j < 4; j++)
						line(result, rect_points[j], rect_points[(j + 1) % 4], lineColor, 2, 8);
				}
				showResult(result);
			}
		}
		return resultPD;
	}
	string CaffeRecognise::process(char* imagebuffer, int size)
	{
		Mat src = imdecode(Mat(1, size, CV_8U, imagebuffer), IMREAD_COLOR);
		if (!src.data)
			return "";
		int result = 0;
		std::vector<CPlate> plateVecOut;
		string resStr = "";
		
		
		result = plateRecognizeMain(src, plateVecOut);
	
		
		if (result == 0)
		{
			size_t num = plateVecOut.size();
			for (size_t j = 0; j < num; j++)
			{
				std::string plateIdentify = plateVecOut[j].getPlateStr();
				resStr += plateIdentify +"\t";

			}
		}
		return resStr;
	}

}