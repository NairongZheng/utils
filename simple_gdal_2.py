"""
    author:nrzheng
    function:use gdal to transform the coordinates
    date:2021.12.2
    reference:https://blog.csdn.net/MLH7M/article/details/120981599
"""
import gdal
import osr

PIC_PATH = r'D:\项目\data\江苏射阳多维度SAR和可见光数据（25平方公里）\S-SAR.tif'

def change_lonlat_dms(lonlat):
    """
        把经纬度转成用度、分、秒表示
        返回的是经纬度字符串
    """
    lon = lonlat[0]
    lat = lonlat[1]

    lon_degree = int(lon)
    lon_minute = int((lon - lon_degree) * 60)
    lon_second = ((lon - lon_degree) * 60 - lon_minute) * 60

    lat_degree = int(lat)
    lat_minute = int((lat - lat_degree) * 60)
    lat_second = ((lat - lat_degree) * 60 - lat_minute) * 60

    lon_str = '{}°{}\'{:.2f}"'.format(lon_degree, lon_minute, lon_second)
    lat_str = '{}°{}\'{:.2f}"'.format(lat_degree, lat_minute, lat_second)
    lonlat_str = str('(') + lon_str + str(', ') + lat_str + str(')') 
    return lonlat_str

def get_reference_sys(dataset):
    """
        获得该图的投影参考系和地理参考系
    """
    proj_rs = osr.SpatialReference()
    proj_rs.ImportFromWkt(dataset.GetProjection())
    geosr_s = proj_rs.CloneGeogCS()
    return proj_rs, geosr_s

def geo2lonlat(dataset, point):
    """
        地理坐标转化成经纬度
        point是一个元组，存放的是地理坐标(x, y)
        返回的也是一个元组，就是该地理坐标转换之后的经纬度
    """
    x = point[0]
    y = point[1]
    proj_rs, geo_rs = get_reference_sys(dataset)
    ct = osr.CoordinateTransformation(proj_rs, geo_rs)
    coords = ct.TransformPoint(x, y)
    return coords[:2]

def imagexy2geo(geo, row_col):
    """
        根据仿射矩阵geo计算图像某点的地理坐标
        row_col是一个元组，存放的是待计算的点与原点的偏移量
        返回的也是一个元组，就是该点的地理坐标(x, y)
    """
    row = row_col[0]
    col = row_col[1]
    px = geo[0] + col * geo[1] + row * geo[2]
    py = geo[3] + col * geo[4] + row * geo[5]
    return (px, py)

def read_img(img_path):
    """
        读取图像信息
    """
    img = gdal.Open(img_path)
    height = img.RasterYSize        # 获取图像的行数
    width = img.RasterXSize         # 获取图像的列数
    band_num = img.RasterCount      # 获取图像波段数

    geo = img.GetGeoTransform()     # 仿射矩阵
    proj = img.GetProjection()      # 地图投影信息，字符串表示

    return img, height, width, band_num, geo, proj

def main():
    (img, height, width, band_num, geo, proj) = read_img(PIC_PATH)
    left_up = (geo[0], geo[3])
    right_up = imagexy2geo(geo, (0, width))
    left_down = imagexy2geo(geo, (height, 0))
    right_down = imagexy2geo(geo, (height, width))

    left_up_lonlat = geo2lonlat(img, left_up)
    right_up_lonlat = geo2lonlat(img, right_up)
    left_down_lonlat = geo2lonlat(img, left_down)
    right_down_lonlat = geo2lonlat(img, right_down)

    left_up_lonlat_ = change_lonlat_dms(left_up_lonlat)
    right_up_lonlat_ = change_lonlat_dms(right_up_lonlat)
    left_down_lonlat_ = change_lonlat_dms(left_down_lonlat)
    right_down_lonlat_ = change_lonlat_dms(right_down_lonlat)

    print('仿射矩阵是:{}'.format(geo))
    print('投影信息是:{}'.format(proj))
    print('左上角的经纬度是:{}, {}'.format(left_up_lonlat, left_up_lonlat_))
    print('右上角的经纬度是:{}, {}'.format(right_up_lonlat, right_up_lonlat_))
    print('左下角的经纬度是:{}, {}'.format(left_down_lonlat, left_down_lonlat_))
    print('右下角的经纬度是:{}, {}'.format(right_down_lonlat, right_down_lonlat_))
    print('图像尺寸是：{}'.format((height, width)))
    pass
    
if __name__ == '__main__':
    main()
