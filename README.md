# utils
一些常用的工具函数（标x的表示比较局限的特定任务用的）
- stores_prints_param.py：训练过程中用来记录和打印信息的方法
- cal_ConfusionMatrix_indicators.py：计算混淆矩阵和语义分割的一些指标
- simple_gdal_1.py：简单的gdal的使用
- simple_gdal_2.py：用gdal计算图像的经纬度
- gengif.m：用文件夹中的图片生成gif图
- plot_histogram.py：遍历文件夹的图片，画出三个通道的直方图
- changelabel_123_imageLabeler.py：把标签从单通道转到三通道（imageLabeler标完的）（x）
- find_edge.py：找到图像的边界，用来找项目那个的白边（x）
- delete_edge.py：把项目图像的白边替换成黑边，还有去除标签超出的边界（x）
