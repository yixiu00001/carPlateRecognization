#include <string>
#include <sstream>
#include "easypr/core/caffe_recognise.h"
using namespace std;
#ifndef _Included_Process
#define _Included_Process

class Process{

private:
	caffepr::CaffeRecognise* cr;
public:
	~Process(){
	}
	Process();
	Process(string);
	string process(char*, int);
};
#endif