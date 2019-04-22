#include <iostream>
#include <string>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>

using namespace std;
using namespace cv;
int local_pict=0;
int success_pict=0;
vector<Mat> image_raw;//储存原始图片
vector<Mat> image_corect;//储存找到角点的图片
vector<vector<Point2f>>  corners_Seq;   //保存检测到的所有角点/
vector<Point2f> corners;                //缓存每幅图像上检测到的角点
struct init_parameter
{
  //棋盘
  int chess_row;
  int chess_col;
  int square_length;//mm

  int picture_number;
  int model;//pin_hole 1;fish_eye 2;
  string picture_src;
  Size board_sz ;
  //const string calibration_file="calibration.txt";
};
void read_parameter(string a[],init_parameter &b)//初始化参数
{
stringstream ss;
ss<<a[0];
ss>>b.chess_row;
ss<<a[1];
ss>>b.chess_col;
ss<<a[2];
ss>>b.square_length;
if (a[3][0]=='f') b.model=2;
else b.model=1;
ss<<a[4];
ss>>b.picture_number;
ss<<a[5];
ss>>b.picture_src;
b.board_sz = Size(b.chess_row, b.chess_col);
}
void print_parameter(init_parameter a)
{

    cout<<"chess_row:"<<a.chess_row<<endl;
    cout<<"chess_col:"<<a.chess_col<<endl;
    cout<<"length:"<<a.square_length<<endl;

    cout<<"model:"<<a.model<<endl;
    cout<<"pict number:"<<a.picture_number<<endl;
    cout<<"pict src:"<<a.picture_src<<endl;
}
int main()
{
    cout<<"enter init_parameter.txt"<<endl;
    string src;
    getline(cin,src);


    //cout<<"init_parameter.txt src"<<src<<endl;
    init_parameter camera;
    //cout<<"calibration src:"<<camera.calibration_file<<endl;
    ifstream readfile;

    string src_txt;
    src_txt=src+"/init_parameter.txt";//初始化参数读入
    readfile.open(src_txt,ios::in);
    if(!readfile){cout<<"error:can't read init_parameter!"<<endl;return 0;}
    string readline[6];
    for(int i=0;i<6;i++)
    {
        cout<<"di "<<i<<"hang:"<<endl;
        getline(readfile,readline[i]);
        cout<<readline[i]<<endl<<endl;
    }
    readfile.close();
    //参数给camera
    read_parameter(readline,camera);

    print_parameter(camera);//打印初始参数


    for(int i=0;i<camera.picture_number;i++)
    {
        string pict_src=camera.picture_src+"left-00"+to_string(i)+".png";//随图片格式改,0 start
        //string pict_src=camera.picture_src+to_string(i+1)+".jpg";//随图片格式改
        Mat p=imread(pict_src,IMREAD_COLOR);
        //cout<<p.size()<<endl;

        image_raw.push_back(p);//放入未标定图片

    }

 while(local_pict<camera.picture_number)//find picture chess corner
 {
 Mat imageGray;
 cvtColor(image_raw[local_pict],imageGray,CV_RGB2GRAY);
 bool patternfound = findChessboardCorners(image_raw[local_pict], camera.board_sz, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE +
                 CALIB_CB_FAST_CHECK);
 printf("patternfound is %d", patternfound);
 if (patternfound)
             {
                 /* 亚像素精确化 */
                 cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
                 /* 绘制检测到的角点并保存 */
                 drawChessboardCorners(image_raw[local_pict], camera.board_sz, corners, patternfound);
                 //imshow("Calibration", image_raw[local_pict]);
                 src_txt=src+"/img_corner/"+"Calibration_"+to_string(success_pict)+".jpg";
                 //cout<<endl<<src_txt<<endl;
                 imwrite(src_txt, image_raw[local_pict]);
                 success_pict++;
                 corners_Seq.push_back(corners);
                 image_corect.push_back(image_raw[local_pict]);
                 //waitKey(500);

                 //destroyWindow("Calibration");
             }


 local_pict++;
 }
 cout << "角点提取完成！" << endl;
 /************************************************************************
 摄像机定标
 *************************************************************************/
 Size square_size = Size(camera.square_length, camera.square_length);
 vector<vector<Point3f>>  object_Points;        /****  保存定标板上角点的三维坐标   ****/

 vector<int>  point_counts;//角点数量
 /* 初始化定标板上角点的三维坐标 */
     for (int t = 0; t<success_pict; t++)
     {
 //		printf("t is %d", t);
         vector<Point3f> tempPointSet;
         for (int i = 0; i<camera.board_sz.height; i++)
         {
 //			printf("i is %d", i);
             for (int j = 0; j<camera.board_sz.width; j++)
             {
 //				printf("j is %d", j);
                 /* 假设定标板放在世界坐标系中z=0的平面上 */
                 Point3f tempPoint;
                 tempPoint.x = i*square_size.width;
                 tempPoint.y = j*square_size.height;
                 tempPoint.z = 0;
                 tempPointSet.push_back(tempPoint);
             }
         }
         object_Points.push_back(tempPointSet);
     }
     cout<<endl<<"test 3d dian!"<<endl<<object_Points[0]<<endl;
     for (int i = 0; i< success_pict; i++)
         {
             point_counts.push_back(camera.board_sz.width*camera.board_sz.height);
         }
     /* 开始定标 */
         Size image_size = image_corect[0].size();
         cv::Matx33d intrinsic_matrix;    /*****    摄像机内参数矩阵    ****/
         cv::Vec4d distortion_coeffs;     /* 摄像机的4个畸变系数：k1,k2,k3,k4*/
         std::vector<cv::Vec3d> rotation_vectors;                           /* 每幅图像的旋转向量 */
         std::vector<cv::Vec3d> translation_vectors;                        /* 每幅图像的平移向量 */
         int flags = 0;
         double err_first;
         if(camera.model==2){
             cout<<endl<<"fisheye calibration!"<<endl;
         flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
         flags |= cv::fisheye::CALIB_CHECK_COND;
         flags |= cv::fisheye::CALIB_FIX_SKEW;
         err_first=fisheye::calibrate(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));
         }
         else {
             cout<<endl<<"pincore calibration!"<<endl;
             flags= CV_CALIB_USE_INTRINSIC_GUESS;
             err_first=calibrateCamera(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));
         }
         cout << "定标完成！\n";
         cout << "重投影误差：" << err_first << "像素" << endl << endl;
         /************************************************************************
             对定标结果进行评价
         *************************************************************************/
             cout << "开始评价定标结果………………" << endl;
             double total_err = 0.0;                   /* 所有图像的平均误差的总和 */
             double err = 0.0;                        /* 每幅图像的平均误差 */
             vector<Point2f>  image_proj2;             /****   保存重新计算得到的投影点    ****/

             cout << "每幅图像的定标误差：" << endl;
             cout << "每幅图像的定标误差：" << endl << endl;
             for (int i = 0; i<success_pict; i++)
             {
                 vector<Point3f> tempPointSet = object_Points[i];
                 /****    通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点     ****/
                 if(camera.model==2) fisheye::projectPoints(tempPointSet, image_proj2, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs);
                 else projectPoints(tempPointSet, image_proj2, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs);
                 /* 计算新的投影点和旧2D检测点之间的误差*/
                 vector<Point2f> tempImagePoint = corners_Seq[i];
                 Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
                 Mat image_proj2Mat = Mat(1, image_proj2.size(), CV_32FC2);
                 for (size_t i = 0; i != tempImagePoint.size(); i++)
                 {
                     image_proj2Mat.at<Vec2f>(0, i) = Vec2f(image_proj2[i].x, image_proj2[i].y);
                     tempImagePointMat.at<Vec2f>(0, i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
                 }
                 err = norm(image_proj2Mat, tempImagePointMat, NORM_L2);
                 total_err += err /= point_counts[i];
                 cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
                 cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
             }
             cout << "总体误差：" << total_err  << "像素" << endl;
             cout << "总体平均误差：" << total_err / success_pict << "像素" << endl << endl;
             cout << "评价完成！" << endl;
    ofstream outfile;
    src_txt=src+"/calibration.txt";
    outfile.open(src_txt);
    outfile<<"重投影误差:"<<err_first<<"像素"<<endl;
    outfile << "总体误差：" << total_err  << "像素" << endl;
    outfile << "总体平均误差：" << total_err / success_pict << "像素" << endl << endl;
    outfile << "内参矩阵"<<intrinsic_matrix<<endl;
    outfile << "D 矩阵:"<<distortion_coeffs<<endl;
    outfile.close();

    /************************************************************************
    显示定标结果
    *************************************************************************/
    Mat mapx = Mat(image_size, CV_32FC1);
    Mat mapy = Mat(image_size, CV_32FC1);
    //这里可以设置R参数
    Mat R = Mat::eye(3, 3, CV_32F);
//    float m0[]={0.9999226574712592, 0.0038404288893699665, 0.011829208830718859,
//       -0.0038676876254582337, 0.9999899158977592, 0.0022823442388144512,
//            -0.011820324363017349, -0.0023279194011530946, 0.9999274277282401};
//    Mat R(3, 3, CV_32F);
//    for(int i=0;i<R.rows;i++)
//      for(int j=0;j<R.cols;j++)
//       R.at<float>(i,j)=*(m0+i*R.rows+j);
    cout<<"R:"<<R<<endl;
    cout << "保存矫正图像" << endl;
    for (int i = 0; i != local_pict; i++)
    {
        Mat t = image_raw[i].clone();
        if(camera.model==2){
        fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);


        }
        else {
            initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);
                    }
        //鱼眼矫正另一种写法
//        Mat intrinsic_mat(intrinsic_matrix), new_intrinsic_mat;
//        intrinsic_mat.copyTo(new_intrinsic_mat);
//        new_intrinsic_mat.at<double>(0,0) *=1.1;
//        new_intrinsic_mat.at<double>(1,1) *=1.1;
//         new_intrinsic_mat.at<double>(0,2) =0.5*image_raw[0].cols;
//        new_intrinsic_mat.at<double>(1,2)=0.5*image_raw[0].rows;
//        std::cout<<"newiintrinsic:"<<new_intrinsic_mat<<std::endl;
//        std::cout<<"fish!"<<std::endl;
//        cv::fisheye::undistortImage(image_raw[local_pict],  t, intrinsic_matrix, distortion_coeffs,new_intrinsic_mat);



        cv::remap(image_raw[i], t, mapx, mapy, INTER_LINEAR);
        src_txt=src+"/img_correct/"+"d_"+to_string(i+1)+".jpg";
        //cout<<endl<<src_txt<<endl;
        imwrite(src_txt, t);
        cout << "img" << i + 1 << "保存" << endl;
    }
    cout << "保存结束" << endl;




    cout << "Hello World!" << endl;
    return 0;
}
