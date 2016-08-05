#ifndef PRCNN_HPP
#define PRCNN_HPP
#include "direct.h" 
#include "io.h"  
#include "easypr\core\caffe_recognise.h"
#include <vector>

namespace easypr {

	namespace demo {

		using namespace cv;
		using namespace std;


		int pr_test_plate_locate()
		{
			cout << "test_plate_locate" << endl;

			const string file = "resources/image/plate_locate.jpg";

			cv::Mat src = imread(file);

			vector<cv::Mat> resultVec;
			CPlateLocate plate;
			//plate.setDebug(0);
			//plate.setLifemode(true);

			int result = plate.plateLocate(src, resultVec);
			if (result == 0) 
			{
				size_t num = resultVec.size();
				for (size_t j = 0; j < num; j++) 
				{
					cv::Mat resultMat = resultVec[j];
					imshow("plate_locate", resultMat);
					waitKey(0);
				}
				destroyWindow("plate_locate");
			}

			return result;
		}

		vector<string> BrowseFilenamesOneLayer(const char* dir, const char *filespec)
		{
			_chdir(dir);
			_finddata_t fileInfo;
			long lfDir;
			//intptr_t  lfDir;
			vector<string> fileVec;
			fileVec.clear();
			if ((lfDir = _findfirst(filespec, &fileInfo)) == -1l)
				cout << "No file is found" << endl;
			else
			{
				do
				{
					char filename[_MAX_PATH];
					strcpy_s(filename, dir);
					strcat_s(filename, "//");
					strcat_s(filename, fileInfo.name);


					fileVec.push_back(filename);

				} while (_findnext(lfDir, &fileInfo) == 0);
			}
			_findclose(lfDir);

			return fileVec;
		}
		int test_plate_locate_batch(const char*path) 
		{
			cout << "test_plate_locate" << endl;
			char* chCurPath = getcwd(NULL, 0);
			vector<cv::Mat> matVec;
			vector<cv::Mat> resultVec;
			CPlateLocate plate;

			vector<string> fileVec;
			cv::Mat src;
			fileVec = BrowseFilenamesOneLayer(path, "*.jpg");
 
			_chdir(chCurPath);
			cout << fileVec.size() << endl;
			for (int i = 0; i < fileVec.size(); i++)
			{
				string fileName = fileVec[i];
				cout << "fileName=" << fileName << endl;
				//const string file = "resources/image/plate_locate.jpg";
				src = imread(fileName);
				int result = plate.plateLocate(src, matVec);
				if (result == 0) 
				{
					size_t num = matVec.size();
					for (size_t j = 0; j < num; j++) 
					{
						cv::Mat resultMat = matVec[j];
						//imshow("plate_locate", resultMat);
						//waitKey(0);
						ostringstream os;
						char a[10];
						itoa(j, a, 10);
						os << "resources/image/general_test_pr/" << a << ".jpg";
						cout << os.str() << endl;
						imwrite(os.str(), resultMat);
					}
					//destroyWindow("plate_locate");
					
				}
			}

			return 0;
		}
		int test_plate_judge(const char* path)
		{
			cout << "test_plate_locate" << endl;
			char* chCurPath = getcwd(NULL, 0);
			vector<string> fileList;
			fileList = BrowseFilenamesOneLayer(path, "*.jpg");
			_chdir(chCurPath);

			vector<cv::Mat> matVec;
			vector<cv::Mat> resultVec;

			for (int i = 0; i < fileList.size();i++)
			{
				string fileName = fileList[i];
				Mat src = imread(fileName);
				matVec.push_back(src);
			}

			int resultJu = PlateJudge::instance()->plateJudge(matVec, resultVec);
			if (resultJu != 0)
			{
				cout << "plateJudge failed!" << endl;
				return -1;
			}
			size_t num = resultVec.size();
			for (size_t j = 0; j < num; j++)
			{
				Mat resMat = resultVec[j];
				ostringstream os;
				char a[10];
				itoa(j, a, 10);
				os << "resources/image/general_test_prLast/" << a << ".jpg";
				cout << os.str() << endl;
				imwrite(os.str(), resMat);
			}
			return 0;

		}
		void scaleIntervalSampling(const Mat &src, Mat &dst, double xRatio, double yRatio)
		{
			//只处理uchar型的像素
			CV_Assert(src.depth() == CV_8U);

			// 计算缩小后图像的大小
			//没有四舍五入，防止对原图像采样时越过图像边界
			int rows = static_cast<int>(src.rows * xRatio);
			int cols = static_cast<int>(src.cols * yRatio);

			dst.create(rows, cols, src.type());

			const int channesl = src.channels();

			switch (channesl)
			{
			case 1: //单通道图像
			{
						uchar *p;
						const uchar *origal;

						for (int i = 0; i < rows; i++){
							p = dst.ptr<uchar>(i);
							//四舍五入
							//+1 和 -1 是因为Mat中的像素是从0开始计数的
							int row = static_cast<int>((i + 1) / xRatio + 0.5) - 1;
							origal = src.ptr<uchar>(row);
							for (int j = 0; j < cols; j++){
								int col = static_cast<int>((j + 1) / yRatio + 0.5) - 1;
								p[j] = origal[col];  //取得采样像素
							}
						}
						break;
			}

			case 3://三通道图像
			{
					   Vec3b *p;
					   const Vec3b *origal;

					   for (int i = 0; i < rows; i++) {
						   p = dst.ptr<Vec3b>(i);
						   int row = static_cast<int>((i + 1) / xRatio + 0.5) - 1;
						   origal = src.ptr<Vec3b>(row);
						   for (int j = 0; j < cols; j++){
							   int col = static_cast<int>((j + 1) / yRatio + 0.5) - 1;
							   p[j] = origal[col]; //取得采样像素
						   }
					   }
					   break;
			}
			}
		}
		int test_plate_detect(const char*path) 
		{
			cout << "test_plate_detect" << endl;
			char* chCurPath = getcwd(NULL, 0);
			vector<cv::Mat> matVec;
			vector<cv::Mat> resultVec;
			CPlateLocate plate;

			vector<string> fileVec;
			cv::Mat src;
			fileVec = BrowseFilenamesOneLayer(path, "*.jpg");

			_chdir(chCurPath);
			cout << fileVec.size() << endl;
			double fScale = 1;
			for (int i = 0; i < fileVec.size(); i++)
			{
				string fileName = fileVec[i];
				cout << "fileName=" << fileName << endl;

				cv::Mat src = imread(fileName);
				int rows = src.rows * fScale;
				int cols = src.cols * fScale;

				cv::Mat dst;
				scaleIntervalSampling(src, dst, fScale, fScale);
				vector<CPlate> resultVec;
				CPlateDetect pd;
				pd.setPDLifemode(true);
				//pd.setDetectShow(true);

				pd.setDetectType(PR_DETECT_COLOR|PR_DETECT_CMSER );
				int result = pd.plateDetect(dst, resultVec);

				int pos;
				string realName;
				pos = fileName.find_last_of('/');
				realName = fileName.substr(pos + 1);
				if (result == 0)
				{
					size_t num = resultVec.size();
					for (size_t j = 0; j < num; j++)
					{
						CPlate resultMat = resultVec[j];
						ostringstream os;
						char a[10];
						itoa(j, a, 10);
						
						os << "resources/image/general_test_prLast2/"<<realName<<"_" << a << ".jpg";
						cout << "tarName="<<os.str() << endl;
						imwrite(os.str(), resultMat.getPlateMat());
						//imshow("plate_detect", resultMat.getPlateMat());
						//waitKey(0);
					}
				}
				else
					cout << "Detect failed!" << endl;
					//destroyWindow("plate_detect");
			}
				return 0;
		}

		int test_chars_segment(const char*path)
		{

			cout << "test_chars_segment" << endl;
			char* chCurPath = getcwd(NULL, 0);
			vector<cv::Mat> matVec;
			vector<cv::Mat> resultVec;
			CPlateLocate plate;

			vector<string> fileVec;
			cv::Mat src;
			fileVec = BrowseFilenamesOneLayer(path, "*.jpg");

			_chdir(chCurPath);
			cout << fileVec.size() << endl;
			int result;
			for (int i = 0; i < fileVec.size(); i++)
			{
				string fileName = fileVec[i];
				string realName,realPath;
				int pos;
				pos = fileName.find_last_of("/");
				realName = fileName.substr(pos + 1);
				ostringstream os;
				os << "resources/image/general_test_prLast3/" << realName ;
				realPath = os.str();
				

				cv::Mat src = imread(fileName);

				vector<Mat> resultVec;
				CCharsSegment plate;
				//"general_test_prLast3"
				result = plate.charsSegment(src, resultVec);
				if (result == 0) 
				{
					size_t num = resultVec.size();
					if (num != 7)
					{
						cout << "Not PR: " << fileName << endl;
						continue;
					}
					_mkdir(realPath.c_str());
					for (size_t j = 0; j < num; j++) 
					{
						cv::Mat resultMat = resultVec[j];
						char a[10];
						itoa(j, a, 10);
						imwrite(realPath+"/" + a + ".jpg", resultMat);
						//imshow("plate_detect", resultMat.getPlateMat());
						//waitKey(0);
					}
					//destroyWindow("plate_detect");
				}
			}
			return result;
		}
		int test_chars_recognisepr() {
			std::cout << "test_chars_recognise" << std::endl;

			cv::Mat src = cv::imread("resources/image/chars_recognise.jpg");
			CCharsRecognise cr;
			
			std::string plateLicense = "";
			int result = cr.charsRecognise(src, plateLicense);
			if (result == 0)
				std::cout << "charsRecognise: " << plateLicense << std::endl;
			return result;
		}
		int test_chars_recognisecaffe() {
			//std::cout << "test_chars_recognise" << std::endl;

			//cv::Mat src = cv::imread("resources/image/plate_judge.jpg");
			
			Mat src = cv::imread("resources/image/7.jpg");
			if (src.empty())
				return -1;
			caffepr::CaffeRecognise cr;
			int result = 0;
			std::string plateIdentify = "";
			//result = cr.recogniseCaffe(src, plateIdentify);
			//cout << "plate=" << plateIdentify << endl;
			
			std::vector<CPlate> plateVecOut;
			result = cr.plateRecognizeMain(src, plateVecOut);
			if (result == 0)
			{
				size_t num = plateVecOut.size();
				for (size_t j = 0; j < num; j++)
				{
					std::string plateIdentify = plateVecOut[j].getPlateStr();
					cout << "Identify=" << plateIdentify << endl;

				}
			}
			
			return result;
		}
		string test_plateRecognizeMain() 
		{
		
			const char* fileName = "resources/image/7.jpg";
			Mat img = cv::imread(fileName);
			if (img.empty())
			{
				cout << "plate is empty" << endl;
				return "";
			}
				
			int size = img.elemSize()*img.total();
			vector<uchar> buff;//buffer for coding
			vector<int> param = vector<int>(2);
			param[0] = CV_IMWRITE_JPEG_QUALITY;
			param[1] = 95;//default(95) 0-100

			imencode(".jpg", img, buff, param);
			uchar *pImg = &(buff[0]);
			char *ppImg = (char*)pImg;

			//Mat src = imdecode(Mat(1, size, CV_8U, pp), IMREAD_COLOR);
	
			
			caffepr::CaffeRecognise cr;
			string res;
		
			res = cr.process(ppImg,size );

			cout << "检测的车牌号为：" <<endl<< res << endl;
			return res;
		}

		int testSvmTrain()
		{
			int result = 0;
			//easypr::SvmTrain svm;
			//svm.train();


			return result;

		}
	}
}

#endif  // EASYPR_PLATE_HPP
