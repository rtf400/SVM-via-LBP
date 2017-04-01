# SVM-via-LBP
Use 59(Uniform LBP) * 68(Facial Landmark) vector in SVM
# Data:
https://drive.google.com/open?id=0B2ETmDSqcgIpNW5jX1RpM0dISzQ
- 400x400images(png) and Landmark(txt) are included in this data. But their landmark is not correct. Just use them for testing
# Set Image & txt fiel Route:

1. Training Data

   >> In main function, you can find "sprintf". It is for training, and you should change this route.
   
2. Testing Data

   >> In "Get_DATA" function, you can find "sprintf". It is for testing, and you should change this route.
   
# Code Procedure:
1. Get LBP Image
   >> "getLBPImage" make LBP Image and get LBP Histogram,
   
2. Get Uniform LBP Histogram
   >> Make Uniform LBP Histogram using "lookup" table. And Store the data in the matrix("trainingDataMat")
   >> And Set the "labelsMat" matrix to distinguish features.
   
3. Train the SVM
   
4. Test sample data
