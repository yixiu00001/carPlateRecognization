# include "../include/Process.h"
# include <sstream>
string Process::process(char* imagebuffer, int size){
	
	string res;
	res = cr->process(imagebuffer, size);
	return res;

	
}

Process::Process(){
	//一些预处理
	cr = new caffepr::CaffeRecognise();
}

Process::Process(string path){
	cr = new caffepr::CaffeRecognise(path);
}