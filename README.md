# utils
一些常用的工具函数（标x的表示比较局限的特定任务用的, *可以看看）
- imageLabel_change_path.m：修改gTruth中的两个变量的路径
- divide_data.py：划分数据集
- stores_prints_param.py：训练过程中用来记录和打印信息的方法
- cal_ConfusionMatrix_indicators.py：计算混淆矩阵和语义分割的一些指标
- cal_ConfusionMatrix_indicators_2.py：计算混淆矩阵和语义分割的一些指标2（用这个）
- simple_gdal_1.py：简单的gdal的使用
- simple_gdal_2.py：用gdal计算图像的经纬度
- gengif.m：用文件夹中的图片生成gif图
---
- plot_histogram.py：遍历文件夹的图片，画出三个通道的直方图
- changelabel_123_imageLabeler.py：把标签从单通道转到三通道（imageLabeler标完的）（x）
- find_edge.py：找到图像的边界，用来找项目那个的白边（x）
- delete_edge.py：把项目图像的白边替换成黑边，还有去除标签超出的边界（x）
- delete_edge.m：同delete_edge.py，其中有的图用py会显示损坏，用m就可以，m真好用啊（x）
- plot_pie.py：统计标签类别的占比，并画饼图
    - 计算类别数量
    - 计算类别占比
    - 用matplotlib画饼图
    - 用pyecharts画饼图
- image_with_mask.py：把标签叠加到图片上去
- generate_test_dataset.py：从图片和标签截取区域作为测试集（x*）
    - zip
    - generate_test_dataset_x.m：还是因为py有的图太大写了损坏，所以有一个用m写（xx）
- cutting_images.py：切图
- results_fusion.py：把所有结果最好的融合
---
- cutting_images_2.py：单独切图用这个
- connecting_images.py：拼图，与cutting_images_2.py相对应
- change_label_321.py：三通道转单通道
- change_label_123.py：单通道转三通道
- gen_public_dataset.py：生成公开数据集
- 14and41.py：一张大图切成4张，或者4拼1.（主要是用来标图的，两个代码放一起了）
---
- gen_hyper_SAR.py：多波段合成"高光谱"SAR
- gen_hyper_SAR_test.py：从"高光谱"SAR中截取测试区
- cutting_hyper_SAR.py：对"高光谱"SAR切图
    - gdal格式和numpy互转
- preprocess_hyper：处理高光谱原始图像, 映射到0-255（x）
