# camera_calibration
pin_hole& fish_eye calibration
需要将要标定的图片放到pic文件夹,
读取图片文件时，主函数中需要改变，改成适合个人图片名称的
需要输入标定初始参数，如init_calibration.txt
运行主函数时需要输入当前init_calibration.txt所在文件夹路径，
同时这个文件夹中需要img_corner和img_correct两个文件夹分别储存角点图片和矫正后图片，
R矩阵需要设为符合自己相机的，pinhole或者不知道就设为eye（3）;
结果如calibration所示

#REQUIRED OpenCV3
