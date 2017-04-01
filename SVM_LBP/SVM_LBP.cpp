#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>


using namespace std;
using namespace cv;
using namespace cv::ml;


const char lookup[256] = {												// 256 dimension -> 59 dimension
	0, 1, 2, 3, 4, 58, 5, 6, 7, 58, 58, 58, 8, 58, 9, 10,
	11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15,
	16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
	17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21,
	22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
	58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
	23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
	24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28,
	29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33,
	58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34,
	58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
	58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35,
	36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40,
	58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41,
	42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46,
	47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57 };


Mat getLBPImage(const Mat& image) {										// Get 256 - Dimension LBP

	int nr = image.rows;
	int nc = image.cols;

	Mat result(image.size(), CV_8U, Scalar(0));



	for (int i = 1; i < nr - 1; i++) {
		const uchar* prevRow = image.ptr<uchar>(i - 1);
		const uchar* currRow = image.ptr<uchar>(i);
		const uchar* nextRow = image.ptr<uchar>(i + 1);

		for (int j = 1; j < nc - 1; j++) {
			uchar resPix = 0;
			const uchar currPix = currRow[j];

			resPix |= (currRow[j - 1] >= currPix) << 7;
			resPix |= (nextRow[j - 1] >= currPix) << 6;
			resPix |= (nextRow[j] >= currPix) << 5;
			resPix |= (nextRow[j + 1] >= currPix) << 4;
			resPix |= (currRow[j + 1] >= currPix) << 3;
			resPix |= (prevRow[j + 1] >= currPix) << 2;
			resPix |= (prevRow[j] >= currPix) << 1;
			resPix |= (prevRow[j - 1] >= currPix);
			result.at<uchar>(i, j) = resPix;
		}
	}
	return result;
}


int *Get_DATA(int idata, float *DATA) {

	char *filename, *imagename;											// Variable for open file & image
	filename = (char*)malloc(sizeof(char) * 200);
	imagename = (char*)malloc(sizeof(char) * 200);
	FILE *fp;

	float *landmark_x, *landmark_y;										// Variable for landmark
	landmark_x = (float *)malloc(sizeof(float) * 68);
	landmark_y = (float *)malloc(sizeof(float) * 68);

	float *histogram;													// Variable for histogram
	histogram = (float*)calloc(sizeof(float), 59);

	int value, i, cnt_landmark, x, y; float max;						// Other variables

																		// Open file & image
	sprintf(filename, "C:\\Users\\USER\\Desktop\\Image Processing\\MyData\\03_LandmarkData\\image_%04d_data.txt", idata + 1);
	fp = fopen(filename, "r");
	for (i = 0; i < 68; i++) {
		fscanf(fp, "%f", &landmark_x[i]);
		fscanf(fp, "%f", &landmark_y[i]);
	}
	fclose(fp);
	sprintf(imagename, "C:\\Users\\USER\\Desktop\\Image Processing\\MyData\\03_LandmarkData\\image_%04d_initialized.png", idata + 1);
	Mat image = imread(imagename);
	free(filename);
	free(imagename);

	Mat gray;															// Get LBP(256 Dimension)
	cvtColor(image, gray, CV_BGR2GRAY);
	Mat result = getLBPImage(gray);

	for (cnt_landmark = 0; cnt_landmark < 68; cnt_landmark++) {			// Get LBP(59 Dimension)
		for (y = (int)((landmark_y[cnt_landmark] - 5 < 0) ? 0 : landmark_y[cnt_landmark] - 5);
			y < (int)((landmark_y[cnt_landmark] + 5 > 400) ? 400 : landmark_y[cnt_landmark] + 5); y++) {
			for (x = (int)((landmark_x[cnt_landmark] - 5 < 0) ? 0 : landmark_x[cnt_landmark] - 5);
				x < (int)((landmark_x[cnt_landmark] + 5 > 400) ? 400 : landmark_x[cnt_landmark] + 5); x++) {
				value = result.at<uchar>(y, x);
				histogram[lookup[value]] += 1;
			}
		}
		max = 0;
		for (i = 0; i < 59; i++) {										// Normalize the Histogram
			if (max < histogram[i])
				max = histogram[i];
		}
		for (i = 0; i < 59; i++) {										// Store the histogram at TrainingDataMat
			DATA[cnt_landmark * 59 + i] = histogram[i] / max;
			histogram[i] = 0;
		}

	}
	free(landmark_x);
	free(landmark_y);
	free(histogram);

	return 0;
}


int main(int, char**)
{
	int numdata_normal = 10;
	int numdata_surprised = 10;
	int numdata_all = numdata_normal + numdata_surprised;				// # of Data

	char *filename, *imagename;											// Variable for open file & image
	filename = (char*)malloc(sizeof(char) * 200);
	imagename = (char*)malloc(sizeof(char) * 200);
	FILE *fp;

	float *landmark_x, *landmark_y;										// Variable for landmark
	landmark_x = (float *)malloc(sizeof(float) * 68);
	landmark_y = (float *)malloc(sizeof(float) * 68);

	float *histogram;													// Variable for histogram
	histogram = (float*)calloc(sizeof(float), 59);

	Mat trainingDataMat(numdata_all, 59 * 68, CV_32FC1, Scalar(0));		// Matrix for Training Data & Label
	Mat labelsMat(numdata_all, 1, CV_32SC1);

	int i, j, idata, cnt_landmark, x, y, value; float max;				// Other variables





	printf("Claculate the Uniform LBP - Normal\n");						//Uniform LBP - Normal

	for (idata = 0; idata < numdata_normal; idata++) {
		labelsMat.at<int>(idata,0) = 0;									// Set labels as 0
		printf("image_%04d....", idata + 1);							// Open file & image
		sprintf(filename, "C:\\Users\\USER\\Desktop\\Image Processing\\MyData\\05_Classified&NumberingData\\Normal\\image_%04d_data.txt", idata + 1);
		fp = fopen(filename, "r");
		for (int i = 0; i < 68; i++) {
			fscanf(fp, "%f", &landmark_x[i]);
			fscanf(fp, "%f", &landmark_y[i]);
		}
		fclose(fp);
		sprintf(imagename, "C:\\Users\\USER\\Desktop\\Image Processing\\MyData\\05_Classified&NumberingData\\Normal\\image_%04d_initialized.png", idata + 1);
		Mat image = imread(imagename);

		Mat gray;														// Get LBP(256 Dimension)
		cvtColor(image, gray, CV_BGR2GRAY);
		Mat result = getLBPImage(gray);

		for (cnt_landmark = 0; cnt_landmark < 68; cnt_landmark++) {		// Get LBP(59 Dimension)
			for (y = (int)((landmark_y[cnt_landmark] - 5 < 0) ? 0 : landmark_y[cnt_landmark] - 5);
				y < (int)((landmark_y[cnt_landmark] + 5 > 400) ? 400 : landmark_y[cnt_landmark] + 5); y++) {
				for (x = (int)((landmark_x[cnt_landmark] - 5 < 0) ? 0 : landmark_x[cnt_landmark] - 5);
					x < (int)((landmark_x[cnt_landmark] + 5 > 400) ? 400 : landmark_x[cnt_landmark] + 5); x++) {
					value = result.at<uchar>(y, x);
					histogram[lookup[value]] += 1;
				}
			}
			max = 0;
			for (i = 0; i < 59; i++) {									// Normalize the Histogram
				if (max < histogram[i])
					max = histogram[i];
			}
			for (i = 0; i < 59; i++) {									// Store the histogram at TrainingDataMat
				trainingDataMat.at<float>(idata, cnt_landmark * 59 + i) = histogram[i] / max;
				histogram[i] = 0;
			}
		}
		printf("Done\n");
	}
	printf("Finish - Normal\n\n");
	
	
	


	printf("Claculate the Uniform LBP - Surprised\n");					//Uniform LBP - Surprised

	for (idata = 0; idata < numdata_surprised; idata++) {
		labelsMat.at<int>(idata+numdata_normal, 0) = 1;					// Set labels as 1
		printf("image_%04d....", idata + 1);							// Open file & image
		sprintf(filename, "C:\\Users\\USER\\Desktop\\Image Processing\\MyData\\05_Classified&NumberingData\\Surprised\\image_%04d_data.txt", idata + 1);
		fp = fopen(filename, "r");
		for (i = 0; i < 68; i++) {
			fscanf(fp, "%f", &landmark_x[i]);
			fscanf(fp, "%f", &landmark_y[i]);
		}
		fclose(fp);
		sprintf(imagename, "C:\\Users\\USER\\Desktop\\Image Processing\\MyData\\05_Classified&NumberingData\\Surprised\\image_%04d_initialized.png", idata + 1);
		Mat image = imread(imagename);

		Mat gray;														// Get LBP(256 Dimension)
		cvtColor(image, gray, CV_BGR2GRAY);
		Mat result = getLBPImage(gray);

		for (cnt_landmark = 0; cnt_landmark < 68; cnt_landmark++) {		// Get LBP(59 Dimension)
			for (y = (int)((landmark_y[cnt_landmark] - 5 < 0) ? 0 : landmark_y[cnt_landmark] - 5);
				y < (int)((landmark_y[cnt_landmark] + 5 > 400) ? 400 : landmark_y[cnt_landmark] + 5); y++) {
				for (x = (int)((landmark_x[cnt_landmark] - 5 < 0) ? 0 : landmark_x[cnt_landmark] - 5);
					x < (int)((landmark_x[cnt_landmark] + 5 > 400) ? 400 : landmark_x[cnt_landmark] + 5); x++) {
					value = result.at<uchar>(y, x);
					histogram[lookup[value]] += 1;
				}
			}
			max = 0;
			for (i = 0; i < 59; i++) {									// Normalize the Histogram
				if (max < histogram[i])
					max = histogram[i];
			}
			for (int i = 0; i < 59; i++) {								// Store the histogram at TrainingDataMat
				trainingDataMat.at<float>(idata + numdata_normal, cnt_landmark * 59 + i) = histogram[i] / max;
				histogram[i] = 0;
			}
		}
		printf("Done\n");
	}
	printf("Finish - Surprised\n\n");
	free(filename);
	free(imagename);
	free(landmark_x);
	free(landmark_y);
	free(histogram);

	



	Ptr<SVM> svm = SVM::create();											// Train the SVM
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(trainingDataMat, ROW_SAMPLE, labelsMat);


	
	


	Mat sampleMat(1, 59 * 68, CV_32F);										// Test the Sample image
	float *DATA;
	float response;
	DATA = (float*)malloc(59 * 68 *sizeof(float));
	for (int j = 0; j < 20; j++) {
		Get_DATA(j, DATA);
		for (int i = 0; i < 59 * 68; i++) {
			sampleMat.at<float>(0, i) = *(DATA + i);
		}
		response = svm->predict(sampleMat);

		printf("\n%f", response);
		if (response == 0)
			printf("\tNormal");
		else
			printf("\tSurprised");
	}
	free(DATA);

	return 0;
}