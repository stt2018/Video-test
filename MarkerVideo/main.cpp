#include <SDKDDKVer.h>
#include <stdio.h>
#include <tchar.h>
#include<iostream>
#include<opencv2\opencv.hpp>
#include<opencv2\video\background_segm.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include<opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

namespace {
	const char* about = "Basic marker detection";
	const char* keys =
		"{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
		"DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
		"DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
		"DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16,"
		"DICT_APRILTAG_16h5=17, DICT_APRILTAG_25h9=18, DICT_APRILTAG_36h10=19, DICT_APRILTAG_36h11=20}"
		"{v        |       | Input from video file, if ommited, input comes from camera }"
		"{ci       | 0     | Camera id if input doesnt come from video (-v) }"
		"{c        |       | Camera intrinsic parameters. Needed for camera pose }"
		"{l        | 0.1   | Marker side lenght (in meters). Needed for correct scale in camera pose }"
		"{dp       |       | File of marker detector parameters }"
		"{r        |       | show rejected candidates too }"
		"{refine   |       | Corner refinement: CORNER_REFINE_NONE=0, CORNER_REFINE_SUBPIX=1,"
		"CORNER_REFINE_CONTOUR=2, CORNER_REFINE_APRILTAG=3}";
}
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
	fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
	fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
	fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
	fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
	fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
	fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
	fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
	fs["minDistanceToBorder"] >> params->minDistanceToBorder;
	fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
	fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
	fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
	fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
	fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
	fs["markerBorderBits"] >> params->markerBorderBits;
	fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
	fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
	fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
	fs["minOtsuStdDev"] >> params->minOtsuStdDev;
	fs["errorCorrectionRate"] >> params->errorCorrectionRate;
	return true;
}




int main(int argc, char *argv[])
{

	CommandLineParser parser(argc, argv, keys);
	parser.about(about);

	if (argc < 2)
	{
		parser.printMessage();
		return 0;
	}//if

	int dictionaryId = parser.get<int>("d");
	bool showRejected = parser.has("r");
	bool estimatePose = parser.has("c");
	float markerLength = parser.get<float>("l");

	Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
	if (parser.has("dp"))
	{
		bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
		if (!readOk)
		{
			cerr << "Invalid detector parameters file" << endl;
			return 0;
		}//if
	}//if

	if (parser.has("refine"))
	{
		//override cornerRefinementMethod read from config file
		detectorParams->cornerRefinementMethod = parser.get<int>("refine");
	}//if
	std::cout << "Corner refinement method (0: None, 1: Subpixel, 2:contour, 3: AprilTag 2): " << detectorParams->cornerRefinementMethod << std::endl;

	Ptr<aruco::Dictionary> dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

	//读入视频
	VideoCapture capture("sc.1.mp4");
	//定义一个Mat变量，用于存储每一帧的图像
	Mat frame;
	//结果
	Mat result;

	//写入视频
	int rate = capture.get(CV_CAP_PROP_FPS);

	VideoWriter writer;
	Mat sizemat;
	capture >> sizemat;
	writer.open("sc-1-out.avi", CV_FOURCC('X', 'V', 'I', 'D'), rate, Size(sizemat.cols, sizemat.rows), true);


	//循环显示每一帧
	while (true)
	{
		//读取当前帧
		capture >> frame;
		//若视频播放完成，退出循环



		if (!capture.isOpened())
		{
			cout << "视频读入失败" << endl;
			break;
		}

		/*int rate = capture.get(CV_CAP_PROP_FPS);
		VideoWriter writer;
		writer.open("output.avi", CV_FOURCC('M', 'J', 'P', 'G'), rate, Size(frame.cols,frame.rows), true);*/

		if (frame.empty())
		{
			break;
		}

		imshow("src", frame);  //显示当前帧
							   //waitKey(30);  //延时30ms

							   //对frame进行检测和画图
		vector< int > ids;
		vector< vector< Point2f > > corners, rejected;

		aruco::detectMarkers(frame, dictionary, corners, ids, detectorParams, rejected);

		frame.copyTo(result);
		if (ids.size() > 0) {
			aruco::drawDetectedMarkers(result, corners, ids);
			//imshow("out", imageCopy);
		}
		writer << result;

		imshow("result",result);
		waitKey(30);
	
	}
	capture.release();
	getchar();
	return 0;
}
