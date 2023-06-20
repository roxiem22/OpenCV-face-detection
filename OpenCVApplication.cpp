// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <queue>
#include <random>
#include <fstream>
#include <cmath>


// daca lucrati cu pixeli obiect negrii
#define FG  255  //obiect = alb
#define BG 0 //fond = negru


int Hist[256];
float mniu = 0.0f, sigma = 0.0f;

wchar_t* projectPath;

bool equalPixels(Vec3b p1, Vec3b p2) {
	return (p1[0] == p2[0]) && (p1[1] == p2[1]) && (p1[2] == p2[2]);
}

int aria(Mat src, Vec3b p) {
	int a = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (equalPixels(p, src.at<Vec3b>(i, j))) {
				a += 1;
			}
		}
	}
	return a;
}

void centruMasa(Mat src, Vec3b p, int& randCentru, int& coloanaCentru) {
	int a = aria(src, p);
	int randuri = 0, coloane = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (equalPixels(p, src.at<Vec3b>(i, j))) {
				randuri += i;
				coloane += j;
			}
		}
	}
	randCentru = randuri / a;
	coloanaCentru = coloane / a;
}

float unghiAxaAlungire(Mat src, Vec3b p, int randCentruGreutate, int colCentruGreutate) {
	int S0 = 0, S1 = 0, S2 = 0;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (equalPixels(p, src.at<Vec3b>(i, j))) {
				S0 += (i - randCentruGreutate) * (j - colCentruGreutate);
				S1 += (j - colCentruGreutate) * (j - colCentruGreutate);
				S2 += (i - randCentruGreutate) * (i - randCentruGreutate);
			}
		}
	}

	return atan2f((float)(2 * S0), (float)(S1 - S2)) / 2;
}

float aspectRatio(Mat src, Vec3b p) {
	int Cmax = 0, Rmax = 0;
	int Cmin = 0x7FFFFFFF, Rmin = 0x7FFFFFFF;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (equalPixels(p, src.at<Vec3b>(i, j))) {
				if (Rmin > i) Rmin = i;
				if (Rmax < i) Rmax = i;
				if (Cmin > j) Cmin = j;
				if (Cmax < j) Cmax = j;
			}
		}
	}

	return (float)(Rmax - Rmin + 1)/(Cmax - Cmin + 1);
}

Mat RGBtoHSV(Mat src) {
	
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);
		float V1, S1, H1;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar B = v3[0];
				uchar G = v3[1];
				uchar R = v3[2];

				float r = (float)R / 255;
				float g = (float)G / 255;
				float b = (float)B / 255;

				float M = max(max(r, g), b);
				float m = min(min(r, g), b);
				float C = M - m;
				V1 = M;

				if (V1 != 0) S1 = C / V1;
				else S1 = 0;
				if (C != 0) {
					if (M == r) H1 = 60 * (g - b) / C;
					if (M == g) H1 = 120 + 60 * (b - r) / C;
					if (M == b) H1 = 240 + 60 * (r - g) / C;
				}
				else
					H1 = 0;
				if (H1 < 0)
					H1 = H1 + 360;

				H.at<uchar>(i, j) = H1 * 255 / 360;
				S.at<uchar>(i, j) = S1 * 255;
				V.at<uchar>(i, j) = V1 * 255;
			}
		}
		return H;
}

void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

Mat dilatare(Mat src) {
	int dj[] = { 1,1,0,-1,-1,-1,0,1 };
	int di[] = { 0,-1,-1,-1,0,1,1,1 };
	Mat dst = src.clone();

	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			if (src.at<uchar>(i, j) == FG) {
				for (int k = 0; k < 8; k++) {
					dst.at<uchar>(i + di[k], j + dj[k]) = FG;
				}
			}
		}
	}

	return dst;

}
Mat dilatareRF(Mat src) {
	int dj[] = { 0,1,0,-1 };
	int di[] = { -1,0,1,0 };
	Mat dst = src.clone();

	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			if (src.at<uchar>(i, j) == FG) {
				for (int k = 0; k < 4; k++) {
					dst.at<uchar>(i + di[k], j + dj[k]) = FG;
				}
			}
		}
	}

	return dst;

}
Mat dilatareN(Mat src, int n) {
	Mat dst, temp;
	temp = src.clone();
	dst = temp.clone();

	for (int i = 0; i < n; i++) {
		dst = dilatare(temp);
		temp = dst.clone();
	}

	return dst;
}

Mat eroziune(Mat src) {
	int dj[] = { 1,1,0,-1,-1,-1,0,1 };
	int di[] = { 0,-1,-1,-1,0,1,1,1 };
	Mat dst = src.clone();

	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			if (src.at<uchar>(i, j) == FG) {
				for (int k = 0; k < 8; k++) {
					if (src.at<uchar>(i + di[k], j + dj[k]) == BG) {
						dst.at<uchar>(i, j) = BG;
						break;
					}
				}
			}
		}
	}

	return dst;

}

Mat eroziuneN(Mat src, int n) {
	Mat dst, temp;
	temp = src.clone();
	dst = temp.clone();

	for (int i = 0; i < n; i++) {
		dst = eroziune(temp);
		temp = dst.clone();
	}

	return dst;
}


void getImgs(Mat imgs[13])
{
	char folderName[MAX_PATH] = "C:/Users/omega/OneDrive - Technical University of Cluj-Napoca/Desktop/proiect pi/OpenCVApplication-VS2022_OCV460_basic/sampleset";
	/*if (openFolderDlg(folderName)==0)
		return nullptr;*/

	char fname[MAX_PATH];
	FileGetter fg(folderName,"jpg");
	
	int i = 0;
	while(fg.getNextAbsFile(fname))
	{
		//citirea imaginii;
		Mat src;
		src = imread(fname);
		
		//adaugare in matricea de imagini
		imgs[i++] = src;
		
	}

}

Mat not(Mat src) {
	Mat dst(src.rows, src.cols, CV_8UC1);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == FG) {
				dst.at<uchar>(i, j) = BG;
			}
			else {
				dst.at<uchar>(i, j) = FG;
			}
		}
	}
	return dst;
}
Mat unionM(Mat a, Mat b) {
	Mat dst= Mat(a.rows, a.cols, CV_8UC1);

	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			if (a.at<uchar>(i, j) == FG || b.at<uchar>(i, j) == FG) {
				dst.at<uchar>(i, j) = FG;
			}
			else {
				dst.at<uchar>(i, j) = BG;
			}
		}
	}

	return dst;
}
Mat and (Mat a, Mat b) {
	Mat dst(a.rows, a.cols, CV_8UC1);

	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			if (a.at<uchar>(i, j) == FG && b.at<uchar>(i, j) == FG) {
				dst.at<uchar>(i, j) = FG;
			}
			else {
				dst.at<uchar>(i, j) = BG;
			}
		}
	}

	return dst;
}
bool equal(Mat a, Mat b) {
	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			if (a.at<uchar>(i, j) != b.at<uchar>(i, j)) {
				return false;
			}
		}
	}

	return true;
}
Mat regionFilling(Mat src,int x, int y) {
	
	Mat dst(src.rows, src.cols, CV_8UC1);

	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			dst.at<uchar>(i, j) = BG;
		}
	}

	dst.at<uchar>(x, y) = FG;

	Mat notSrc = not(src);

	Mat newVal;
	newVal = dst;
	do {
		dst = newVal;
		Mat r = dilatareRF(dst);
		newVal = and (r, notSrc);
		
	} while (!equal(dst, newVal));

	return not(newVal);
}

Mat etichetare(Mat src) {

		src = not(src);
		Mat labels = Mat::zeros(src.size(), CV_16UC1);
		Mat dst = Mat::zeros(src.size(), CV_8UC3);
		
		Scalar colorLUT[1000] = { 0 };
		Scalar color;
		for (int i = 0; i < 1000; i++)
		{
			color = Scalar(rand() & 255, rand() & 255, rand() & 255);
			colorLUT[i] = color;
		}

		colorLUT[0] = Scalar(0, 0, 0);

		std::queue<Point> que;
		int label = 0;
		int dj[8] = { 1,1,0,-1,-1,-1,0,1 };
		int di[8] = { 0,-1,-1,-1,0,1,1,1 };

		for (int i = 0; i < labels.rows - 1; i++)
		{
			for (int j = 0; j < labels.cols - 1; j++)
			{
				if (src.at<uchar>(i, j) == 0 && labels.at<ushort>(i, j) == 0)
				{
					que.push(Point(j, i));
					label++;
					labels.at<ushort>(i, j) = label;
					while (!que.empty())
					{
						Point oldest = que.front();
						int ii = oldest.y;
						int jj = oldest.x;
						que.pop();
						for (int k = 0; k < 8; k++)
						{
							if (ii + di[k] >= 0 && ii + di[k] < labels.rows && jj + dj[k] >= 0 && jj + dj[k] < labels.cols) //isInside
								if (src.at<uchar>(ii + di[k], jj + dj[k]) == 0 && labels.at<ushort>(ii + di[k], jj + dj[k]) == 0)
								{
									labels.at<ushort>(ii + di[k], jj + dj[k]) = label;
									que.push(Point(jj + dj[k], ii + di[k]));
								}
						}
					}

					for (int i = 0; i < labels.rows - 1; i++)
					{
						for (int j = 0; j < labels.cols - 1; j++)
						{
							Scalar color = colorLUT[labels.at<ushort>(i, j)];
							dst.at<Vec3b>(i, j)[0] = color[0];
							dst.at<Vec3b>(i, j)[1] = color[1];
							dst.at<Vec3b>(i, j)[2] = color[2];
						}
					}
				}
			}
		}
		
		return dst;
}


void train() {

	Mat imagini[13];
	getImgs(imagini);
	int histL[256] = {0};
	
	for (int i = 0; i < 256; i++) {
		Hist[i] = 0;
	}

	sigma = 0.0f;
	mniu = 0.0f;

	//calc histogrmei
	for (int i = 0; i < 13; i++) {

		Mat H = RGBtoHSV(imagini[i]);

		for (int i = 0; i < H.rows; i++) {
			for (int j = 0; j < H.cols; j++) {
				histL[H.at<uchar>(i, j)] ++;
			}
		
		}
		 
		for (int j = 0; j < 256; j++) {
			Hist[j] += histL[j];
			histL[j] = 0;
		}
		
	}
	
	int maxx = 0;
	for (int j = 0; j < 256; j++)
		if (Hist[j] > maxx)
			maxx = Hist[j];

	for (int j = 0; j < 256; j++)
		if (Hist[j] < maxx * 0.1f)
			Hist[j] = 0;


	//showHistogram("histograma globala", Hist, 256, 200);

	float p[256] = { 0.0f };
	int M = 0;
	int sum = 0;

	for (int j = 0; j < 256; j++)
	{
		sum += Hist[j] * j;
		M += Hist[j];
	}
	
	mniu = (float) sum / M;

	for (int j = 0; j < 256; j++)
	{
		sigma += (j - mniu) * (j - mniu) * Hist[j];
	}
	sigma = sigma / M;
	sigma = sqrtf(sigma);
	
	printf(" Media: %.2f si covariantul: %.2f\n\n\n", mniu, sigma);
	//waitKey();
}

Mat stergere(Mat src,Vec3b color) {
	Mat dst = src.clone();
	for (int i = 1; i < src.rows-1; i++) {
		for (int j = 1; j < src.cols-1; j++) {
			if (src.at<Vec3b>(i, j)[0] == color[0] && src.at<Vec3b>(i, j)[1] == color[1] && src.at<Vec3b>(i, j)[2] == color[2]) {
				
				dst.at<Vec3b>(i, j)[0] = 0;
				dst.at<Vec3b>(i, j)[1] = 0;
				dst.at<Vec3b>(i, j)[2] = 0;
			}
		}
	}
	
	return dst;

}

void likelihood() {
	Mat src = imread("C:/Users/omega/OneDrive - Technical University of Cluj-Napoca/Desktop/proiect pi/OpenCVApplication-VS2022_OCV460_basic/testset/family.jpg", IMREAD_COLOR);	// Read the image
	
	Mat hsv;
	Mat lh;
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	imshow("src", src);
	GaussianBlur(src, src, Size(5, 5), 0, 0);
	Mat channels[] = { Mat::zeros(src.size(), CV_8UC1), Mat::zeros(src.size(), CV_8UC1), Mat::zeros(src.size(), CV_8UC1) };
	cvtColor(src, hsv,COLOR_BGR2HSV);
	split(hsv, channels);
	Mat H;
	H = channels[0] * 255/180;
	float k1 = 2.0;
	int hue_min = mniu - k1 * sigma;
	if (hue_min < 0) hue_min = 0;
	int hue_max = mniu + k1 * sigma;
	if (hue_max > 255) hue_max = 255;
	for (int i = 0; i < H.rows; i++)
	{
		for (int j = 0; j < H.cols; j++)
		{
			uchar hue = H.at<uchar>(i, j);
			if (hue_min <= hue && hue <= hue_max)
				dst.at<uchar>(i, j) = 255;
			else
				dst.at<uchar>(i, j) = 0;
		}
	}
	
	imshow("likelyhood", dst);

	lh = dst.clone();

	Mat fr_s = lh.clone();

	Mat er_di = eroziuneN(fr_s, 4);
	er_di = dilatareN(er_di, 8);
	er_di = eroziuneN(er_di, 4);

	Mat andDiErBin = er_di;
	imshow("Er_Dil", er_di);
	//imshow("AndDiErBin", andDiErBin);*/
	Mat et_fr = etichetare(andDiErBin);
	imshow("Etichetare", et_fr);
	Vec3b x[20];
	int flag = 1;

	for (int l = 0; l < 20; l++) {
		x[l][0] = 0;
		x[l][1] = 0;
		x[l][2] = 0;
	}

	Mat alungire=et_fr.clone();
	int areas[10] = { 0 };
	float angles[10] = { 0.0f };
	int ccc = 0;
	for (int i = 1; i < et_fr.rows-1; i++) {
		for (int j = 1; j < et_fr.cols-1; j++) {
			flag = 1;
			if (et_fr.at<Vec3b>(i, j)[0] != 0 && et_fr.at<Vec3b>(i, j)[1] != 0 && et_fr.at<Vec3b>(i, j)[2] != 0){
				for (int l = 0; l < 20; l++) {
					if (et_fr.at<Vec3b>(i, j)[0] == x[l][0] && et_fr.at<Vec3b>(i, j)[1] == x[l][1] && et_fr.at<Vec3b>(i, j)[2] == x[l][2])
					flag = 0;
				}

				if (flag == 1) {
					for (int l = 0; l < 20; l++) {
						if (x[l][0] == 0 && x[l][1] == 0 && x[l][2] == 0)
						{
							x[l][0] = et_fr.at<Vec3b>(i, j)[0];
							x[l][1] = et_fr.at<Vec3b>(i, j)[1];
							x[l][2] = et_fr.at<Vec3b>(i, j)[2];
							l=20;
						}
					}

					float ar = aspectRatio(et_fr, et_fr.at<Vec3b>(i, j));
					printf("%.2f   %d %d\n", ar,i,j);
					
					if (!(ar > 1 && ar < 3.5)) { 
						alungire = stergere(alungire, et_fr.at<Vec3b>(i, j));
						areas[ccc] = aria(et_fr, et_fr.at<Vec3b>(i, j));
						int centruMasaRand, centruMasaColoana;
						centruMasa(et_fr, et_fr.at<Vec3b>(i, j), centruMasaRand, centruMasaColoana);
						angles[ccc] = unghiAxaAlungire(et_fr, et_fr.at<Vec3b>(i, j), centruMasaRand, centruMasaColoana);
						ccc++;
					}
				}
			}
		}
	}
	Vec3b cul = Vec3b(0,0,0);
	Mat alungireGri = Mat(alungire.rows, alungire.cols, CV_8UC1);
	for (int i = 0; i < alungire.rows; i++) {
		for (int j = 0; j < alungire.cols; j++) {
			if (!equalPixels(alungire.at<Vec3b>(i, j), cul)) {
				alungireGri.at<uchar>(i, j) = 255;

			}
			else
				alungireGri.at<uchar>(i, j) = 0;
		}
	}


	for (int i = 0; i < ccc; i++) {
		printf("Dimensiune: %d Unghiul Axei de alungire: %.3f\n", areas[i], angles[i]);
	}

	imshow("Etichetarebin", alungireGri);
	waitKey(0);
}

void proiect() {
	
	train();
	likelihood();

}


int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);
	proiect();

    
	return 0;
}