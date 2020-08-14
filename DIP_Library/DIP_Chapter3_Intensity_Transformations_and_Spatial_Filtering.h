//	Tutor: Tran Tien Duc
//	Gmail: trantienduc@gmail.com
//	Created by Cao Le Cong Minh
//	Gmail: caolecongminh1997@gmail.com
//	Github: https://github.com/Minh-CaoLeCong

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// the number of possible intensity levels in the image (256 for an 8-bit image)
#define L 256

namespace DIP
{
	class DIP_Chapter3
	{
	public:
		static void Negative(Mat imgin, Mat imgout);
		static void Logarit(Mat imgin, Mat imgout);
		static void Histogram(Mat imgin, Mat imgout);
		static void HistogramEqualization(Mat imgin, Mat imgout);
		static void HistogramSpecification(Mat imgin, Mat imgout);
		static void LocalHistogram(Mat imgin, Mat imgout);
		static void Power(Mat imgin, Mat imgout);
		static void PiecewiseLinear(Mat imgin, Mat imgout);
		static void MyFilter2D(Mat imgin, Mat imgout);
		static void GreyWorld_Algorithm(Mat imgin, Mat imgout);
		static void Histogram_RGB(Mat imgin);
		static void equalizeIntensity(Mat imgin, Mat imgout);
	};
}