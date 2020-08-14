//	Tutor: Tran Tien Duc
//	Gmail: trantienduc@gmail.com
//	Created by Cao Le Cong Minh
//	Gmail: caolecongminh1997@gmail.com
//	Github: https://github.com/Minh-CaoLeCong

#include "DIP_Chapter3_Intensity_Transformations_and_Spatial_Filtering.h"

namespace DIP
{
	/// negative transformation function: s = L - 1 -r
	/* NOTE: denote the values of pixels, before and after processing, by r and s, respectively */
	void DIP_Chapter3::Negative(Mat imgin, Mat imgout)
	{
		// check channels of image 
		// if not grayscale image, convert it 
		if (imgin.channels() >= 3)
			// convert to grayscale image
			cvtColor(imgin, imgin, CV_RGB2GRAY);

		// take the height and width of image
		int M = imgin.size().height;
		int N = imgin.size().width;

		int x, y;
		// NOTE: denote the values of pixels, before and after processing, by r and s, respectively
		uchar r, s;
		
		// throughout each pixel of image 
		for (x = 0; x < M; x++)
			for (y = 0; y < N; y++)
			{
				// take the current pixel value
				r = imgin.at<uchar>(x, y);
				// negative transformation function: s = L - 1 - r
				s = L - 1 - r;
				imgout.at<uchar>(x, y) = s;
			}
		return;

	} // negative transformation function: s = L - 1 -r


	/// the general form of the log transformation: s = c*log(1 + r)
	/* NOTE: denote the values of pixels, before and after processing, by r and s, respectively
		assumed r >= 0
		'c' is a positive constant:
			if r = 0,		then s = 0
		and	if r = L - 1,	then s = L - 1
		so from the form of the log transformation: s = c*log(1 + r)
		we can compute: c = (L - 1) / log(1 + L - 1)
		L = 256 => c = 45.9859																	*/
	void DIP_Chapter3::Logarit(Mat imgin, Mat imgout)
	{

		// check channels of image
		// if not grayscale image, convert it
		if (imgin.channels() >= 3)
			cvtColor(imgin, imgin, CV_RGB2GRAY);

		// take the height and width of image
		int M = imgin.size().height;
		int N = imgin.size().width;

		int x, y;

		// NOTE: denote the values of pixels, before and after processing, by r and s, respectively
		double r, s;
		// c is a positive constant :
		// if r = 0, then s = 0
		//	and if r = L - 1, then s = L - 1
		//	so from the form of the log transformation : s = c * log(1 + r)
		//	we can compute : c = (L - 1) / log(1 + L - 1)
		//	L = 256 = > c = 45.9859
		double c = (L - 1) / log((double)L); // because 'L' was defined by 256, so need to casting to 'double'

		// throughout each pixel of image
		for (x = 0; x < M; x++)
			for (y = 0; y < N; y++)
			{
				// take the current pixel value
				r = imgin.at<uchar>(x, y);
				if (r == 0)
					r = 1;

				// the log transformation: s = c*log(1 + r)
				s = c * log(1 + r);
				imgout.at<uchar>(x, y) = (uchar)s; // casting 's' from 'double' to 'uchar'
			}
		return;

	} // the log transformation: s = c*log(1 + r)


	/// the form of Power-law transformations: s = c * pow(r, gamma)
	/* NOTE: denote the values of pixels, before and after processing, by r and s, respectively
		'c' and 'gamma' is a positive constant.
		we can see:
			if 'gamma' < 1, make the image brighter
			if 'gamma' > 1, make the image darker

			if r = 0,		then s = 0
		and	if r = L - 1,	then s = L - 1
		so the form of Power-law transformations: s = c * pow(r, gamma)
		we can compute: c = (L - 1) / pow(L - 1, gamma) or c = pow(L - 1, 1 - gamma)		*/
	void DIP_Chapter3::Power(Mat imgin, Mat imgout)
	{

		// check channels of image
		// if not grayscale image, convert it
		if (imgin.channels() >= 3)
			cvtColor(imgin, imgin, CV_RGB2GRAY);

		// take the height and width of image
		int M = imgin.size().height;
		int N = imgin.size().width;

		int x, y;

		// NOTE: denote the values of pixels, before and after processing, by r and s, respectively
		double r, s, c;

		// if 'gamma' < 1, make the image brighter
		// if 'gamma' > 1, make the image darker
		double gamma = 0.5;

		// if r = 0, then s = 0
		//	and if r = L - 1, then s = L - 1
		//	so the form of Power - law transformations : s = c * pow(r, gamma)
		//	we can compute : c = (L - 1) / pow(L - 1, gamma) or c = pow(L - 1, 1 - gamma)
		c = pow(L - 1, 1 - gamma);

		// throughout each pixel of image
		for (x = 0; x < M; x++)
			for (y = 0; y < N; y++)
			{
				// take the current pixel value
				r = imgin.at<uchar>(x, y);
				if (r == 0)
					r = 1;
				// Power-law transformations: s = c * pow(r, gamma)
				s = c * pow(r, gamma);
				imgout.at<uchar>(x, y) = (uchar)s;
			}

		return;

	} //  Power-law transformations: s = c * pow(r, gamma)

	void DIP_Chapter3::PiecewiseLinear(Mat imgin, Mat imgout)
	{
		// check channels of image
		// if not grayscale image, convert it
		if (imgin.channels() >= 3)
			cvtColor(imgin, imgin, CV_RGB2GRAY);

		// 'rmin' and 'rmax' denote the the minimum and maximum intensity
		// levels in the input image
		double rmin, rmax;
		// Using 'minMaxLoc' function in OpenCV to finds the minimum 
		// and maximum element values in the input image
		minMaxLoc(imgin, &rmin, &rmax);

		// take the height and width of the input image
		int M = imgin.size().height;
		int N = imgin.size().width;

		int x, y;

		// NOTE: denote the values of pixels, before and 
		// after processing, by r and s, respectively
		double r, s, r1, s1, r2, s2;

		// contrast stretching: 
		//	setting (r1, s1) = (rmin, 0) and (r2, s2) = (rmax, L - 1)
		// thresholding function:
		//	setting (r1, s1) = (m, 0) and (r2, s2) = (m, L - 1)
		//	where m is the mean intensity level in the image
		r1 = rmin; s1 = 0;
		r2 = rmax; s2 = L - 1;

		// throughout each pixel value of the input image
		for (x = 0; x < M; x++)
			for (y = 0; y < N; y++)
			{
				// take the current pixel value
				r = imgin.at<uchar>(x, y);

				if (r < r1)
					s = s1 / r1 * r;
				else if (r < r2)
					s = (s2 - s1) / (r2 - r1) * (r - r1) + s1;
				else
					s = (L - 1 - s2) / (L - 1 - r2) * (r - r2) + s2;

				imgout.at<uchar>(x, y) = (uchar)s;

			}

		return;
	}

	/// the "normalized histogram" is defined as: 
	//	(the probability of occurrence of intensity level 'rk' in a image)
	//	where rk = [0, L - 1]
	///		p(rk) = h(rk) / M * N = nk / M * N
	// NOTE: the sum of p(rk) for all values of k is always 1
	// h(rk) = nk: 
	//	where k = 0, 1, 2, ..., L - 1, denote the intensities of an 
	//		L-level digital image.
	//	'nk' is the number of pixels that have intensity 'rk'.
	// M and N are the number of image rows and columns.
	void DIP_Chapter3::Histogram(Mat imgin, Mat imgout)
	{
		// check channels of image
		// if not grayscale image, convert it
		if (imgin.channels() >= 3)
			cvtColor(imgin, imgin, CV_RGB2GRAY);

		// take the height and width of the input image
		int M = imgin.size().height;
		int N = imgin.size().width;
		
		int x, y;
		int r, h[L];

		for (r = 0; r < L; r++)
			h[r] = 0;

		for (x = 0; x < M; x++)
			for (y = 0; y < N; y++)
			{
				r = imgin.at<uchar>(x, y);
				h[r]++;
			}

		double p[L];

		// M and N are the number of image rows and columns
		for (r = 0; r < L; r++)
			p[r] = (double)h[r] / (M * N);

		int scale = 5000;

		// display histogram
		for (r = 0; r < L; r++)
			line(imgout, Point(r, M - 1), Point(r, M - 1 - (int)(scale * p[r])), CV_RGB(0, 0, 0));
		return;
	}
	
	void DIP_Chapter3::HistogramEqualization(Mat imgin, Mat imgout)
	{
		// check channels of image
		// if not grayscale image, convert it
		if (imgin.channels() >= 3)
			cvtColor(imgin, imgin, CV_RGB2GRAY);

		// take the height and width of the input image
		int M = imgin.size().height;
		int N = imgin.size().width;

		int x, y;

		int r, h[L];
		
		for (r = 0; r < L; r++)
			h[r] = 0;

		for (x = 0; x < M; x++)
			for (y = 0; y < N; y++)
			{
				r = imgin.at<uchar>(x, y);
				h[r]++;
			}

		double p[L];

		for (r = 0; r < L; r++)
			p[r] = (double)h[r] / (M * N);

		double s[L];

		int j, k;

		for (k = 0; k < L; k++)
		{
			s[k] = 0;
			for (j = 0; j <= k; j++)
			{
				s[k] += p[j];
			}
			s[k] *= (L - 1);
		}
		for (x = 0; x < M; x++)
			for (y = 0; y < N; y++)
			{
				r = imgin.at<uchar>(x, y);
				imgout.at<uchar>(x, y) = (uchar)s[r];
			}
		return;
	}

	void DIP_Chapter3::HistogramSpecification(Mat imgin, Mat imgout)
	{
		// check channels of image
		// if not grayscale image, convert it
		if (imgin.channels() >= 3)
			cvtColor(imgin, imgin, CV_RGB2GRAY);

		// take the height and width of the input image
		int M = imgin.size().height;
		int N = imgin.size().width;

		double pz[L];
		double G[L];
		double pr[L];
		double T[L];
		double sum;

		int z, k, i, j, x, y;

		// initially histogram specification
		double pz1, pz2, pz3, pz4, pz5, pz6;
		int z1, z2, z3, z4, z5, z6;
		z1 = 0;		pz1 = 0.75;
		z2 = 10;	pz2 = 7;
		z3 = 20;	pz3 = 0.75;
		z4 = 180;	pz4 = 0;
		z5 = 200;	pz5 = 0.7;
		z6 = 255;	pz6 = 0;
		for (z = 0; z < L; z++)
		{
			if (z < z2)
				pz[z] = (pz2 - pz1) / (z2 - z1)*(z - z1) + pz1;
			else if (z < z3)
				pz[z] = (pz3 - pz2) / (z3 - z2)*(z - z2) + pz2;
			else if (z < z4)
				pz[z] = (pz4 - pz3) / (z4 - z3)*(z - z3) + pz3;
			else if (z < z5)
				pz[z] = (pz5 - pz4) / (z5 - z4)*(z - z4) + pz4;
			else
				pz[z] = (pz6 - pz5) / (z6 - z5)*(z - z5) + pz5;
		}
		sum = 0;
		for (z = 0; z < L; z++)
			sum += pz[z];
		for (z = 0; z < L; z++)
			pz[z] = pz[z] / sum;
		for (k = 0; k < L; k++) 
		{
			G[k] = 0;
			for (i = 0; i <= k; i++)
				G[k] += pz[i];
		}

		// histogram of input image
		int r, h[L];
		for (r = 0; r < L; r++)
			h[r] = 0;
		for (x = 0; x < M; x++)
			for (y = 0; y < N; y++)
			{
				r = imgin.at<uchar>(x, y);
				h[r]++;
			}
		for (r = 0; r < L; r++)
			pr[r] = (double)h[r] / (M * N);
		for (k = 0; k < L; k++)
		{
			T[k] = 0;
			for (j = 0; j <= k; j++)
				T[k] += pr[j];
		}

		// matching histograms
		double s;
		for (x = 0; x < M; x++)
			for (y = 0; y < N; y++) 
			{
				r = imgin.at<uchar>(x, y);
				s = T[r];
				for (k = 0; k < L; k++)
					if (G[k] >= s)
						break;
				imgout.at<uchar>(x, y) = k;
			}
		return;
	}


	void DIP_Chapter3::LocalHistogram(Mat imgin, Mat imgout)
	{
		int m = 3, n = 3;

		Mat win = Mat(m, n, CV_8UC1);
		Mat wout = Mat(m, n, CV_8UC1);

		int M = imgin.size().height;
		int N = imgin.size().width;

		int x, y, s, t;

		int a = m / 2, b = n / 2;

		for (x = a; x < M - a; x++)
			for (y = b; y < N - b; y++)
			{
				for (s = -a; s <= a; s++)
					for (t = -b; t <= b; t++)
						win.at<uchar>(s + a, t + b) = imgin.at<uchar>(x + s, y + t);

				equalizeHist(win, wout);
				imgout.at<uchar>(x, y) = wout.at<uchar>(a, b);
			}
		return;
	}

	void DIP_Chapter3::MyFilter2D(Mat imgin, Mat imgout)
	{
		Mat kernel = Mat::ones(3, 3, CV_8UC1);

		int m = kernel.size().height;
		int n = kernel.size().width;

		int M = imgin.size().height;
		int N = imgin.size().width;

		int x, y, s, t;

		int a = m / 2, b = n / 2;

		float r;

		for (x = a; x < M - a; x++)
			for (y = b; y < N - b; y++)
			{
				r = 0;

				for (s = -a; s <= a; s++)
					for (t = -b; t <= b; t++)
					{
						r += kernel.at<uchar>(s + a, t + b) * imgin.at<uchar>(x + s, y + t);

					}
				imgout.at<uchar>(x, y) = (uchar)r;
			}

		return;
	}

	void DIP_Chapter3::GreyWorld_Algorithm(Mat imgin, Mat imgout)
	{
		// Sum the colour values in each channels
		Scalar Sum_Color_Values_Img = sum(imgin);

		// Normalise by the number of pixels in the image to obtain an estimate for the illuminant
		Scalar illum = Sum_Color_Values_Img / (imgin.rows * imgin.cols);

		//Split the image into different channels
		vector<Mat> RGB_Channels(3);
		split(imgin, RGB_Channels);

		// Assign the three colour channels to CV:Mat variables for processing
		Mat Red_Channel = RGB_Channels[2];
		Mat Green_Channel = RGB_Channels[1];
		Mat Blue_Channel = RGB_Channels[0];

		// Calculate scale factor for normalisation you can use 255 instead
		double scale = (illum(0) + illum(1) + illum(2)) / 3;

		//Correct for illuminant (white balancing)
		Red_Channel = Red_Channel * scale / illum(2);
		Green_Channel = Green_Channel * scale / illum(1);
		Blue_Channel = Blue_Channel * scale / illum(0);

		// Assign the processed channels back into vector to use the cv::merge() function
		RGB_Channels[0] = Blue_Channel;
		RGB_Channels[1] = Green_Channel;
		RGB_Channels[2] = Red_Channel;

		merge(RGB_Channels, imgout);

		return;
	}

	void DIP_Chapter3::Histogram_RGB(Mat imgin)
	{
		//Separate the image into 3 colour channel (R, G and B)
		vector<Mat> RGB_Channels;
		split(imgin, RGB_Channels);

		///Establish the number of bins
		int HistSize = 256;

		///Set the ranges (for R, G, B)
		float range[] = { 0, 256 };
		const float* HistRange = { range };

		bool uniform = true;
		bool accumulate = false;

		Mat Red_Hist, Green_Hist, Blue_Hist;

		//Compute the histograms:
		calcHist(&RGB_Channels[0], 1, 0, Mat(), Blue_Hist, 1, &HistSize, &HistRange, uniform, accumulate);
		calcHist(&RGB_Channels[1], 1, 0, Mat(), Green_Hist, 1, &HistSize, &HistRange, uniform, accumulate);
		calcHist(&RGB_Channels[2], 1, 0, Mat(), Red_Hist, 1, &HistSize, &HistRange, uniform, accumulate);

		// Draw the histograms for B, G and R
		int hist_w = 512; int hist_h = 400;
		int bin_w = cvRound((double)hist_w / HistSize);

		Mat RGB_Histogram(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
		Mat Red_Histogram(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
		Mat Green_Histogram(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
		Mat Blue_Histogram(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

		// Normalize the result to [ 0, histImage.rows ]
		normalize(Blue_Hist, Blue_Hist, 0, RGB_Histogram.rows, NORM_MINMAX, -1, Mat());
		normalize(Green_Hist, Green_Hist, 0, RGB_Histogram.rows, NORM_MINMAX, -1, Mat());
		normalize(Red_Hist, Red_Hist, 0, RGB_Histogram.rows, NORM_MINMAX, -1, Mat());

		/// Draw for each channel
		for (int i = 1; i < HistSize; i++)
		{
			line(RGB_Histogram, Point(bin_w*(i - 1), hist_h - cvRound(Blue_Hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(Blue_Hist.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);
			line(Blue_Histogram, Point(bin_w*(i - 1), hist_h - cvRound(Blue_Hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(Blue_Hist.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);
			line(RGB_Histogram, Point(bin_w*(i - 1), hist_h - cvRound(Green_Hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(Green_Hist.at<float>(i))),
				Scalar(0, 255, 0), 2, 8, 0);
			line(Green_Histogram, Point(bin_w*(i - 1), hist_h - cvRound(Green_Hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(Green_Hist.at<float>(i))),
				Scalar(0, 255, 0), 2, 8, 0);
			line(RGB_Histogram, Point(bin_w*(i - 1), hist_h - cvRound(Red_Hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(Red_Hist.at<float>(i))),
				Scalar(0, 0, 255), 2, 8, 0);
			line(Red_Histogram, Point(bin_w*(i - 1), hist_h - cvRound(Red_Hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(Red_Hist.at<float>(i))),
				Scalar(0, 0, 255), 2, 8, 0);
		}

		namedWindow("RGB_Histogram", WINDOW_AUTOSIZE);
		imshow("RGB_Histogram", RGB_Histogram);

		namedWindow("Red_Histogram", WINDOW_AUTOSIZE);
		imshow("Red_Histogram", Red_Histogram);

		namedWindow("Green_Histogram", WINDOW_AUTOSIZE);
		imshow("Green_Histogram", Green_Histogram);

		namedWindow("Blue_Histogram", WINDOW_AUTOSIZE);
		imshow("Blue_Histogram", Blue_Histogram);

		return;
	}

	void DIP_Chapter3::equalizeIntensity(Mat imgin, Mat imgout)
	{
		if (imgin.channels() >= 3)
		{
			Mat ycrcb;

			cvtColor(imgin, ycrcb, CV_BGR2YCrCb);

			vector<Mat> channels;
			split(ycrcb, channels);

			equalizeHist(channels[0], channels[0]);

			merge(channels, ycrcb);

			cvtColor(ycrcb, imgout, CV_YCrCb2BGR);
		}
		return;
	}
}