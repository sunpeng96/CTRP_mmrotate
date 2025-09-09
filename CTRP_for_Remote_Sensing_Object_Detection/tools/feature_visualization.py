import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

def draw_feature_map1(features, img_path, save_dir = './work_dirs/feature_map/',name = None):
    '''
    :param features: 特征层。可以是单层，也可以是一个多层的列表
    :param img_path: 测试图像的文件路径
    :param save_dir: 保存生成图像的文件夹
    :return:
    '''
    img = cv2.imread(img_path)      #读取文件路径
    i=0
    if isinstance(features,torch.Tensor):   # 如果是单层
        features = [features]       # 转为列表
    for featuremap in features:     # 循环遍历
        heatmap = featuremap_2_heatmap1(featuremap)	#主要是这个，就是取特征层整个的求和然后平均，归一化
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap0 = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
        heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
        superimposed_img = heatmap * 0.5 + img  # 这里的0.4是热力图强度因子
        plt.imshow(heatmap0)  # ,cmap='gray' ，这里展示下可视化的像素值
        # plt.imshow(superimposed_img)  # ,cmap='gray'
        plt.close()	#关掉展示的图片
        # 下面是用opencv查看图片的
        # cv2.imshow("1",superimposed_img)
        # cv2.waitKey(0)     #这里通过安键盘取消显示继续运行。
        # cv2.destroyAllWindows()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, name + str(i) + '.png'), superimposed_img) #superimposed_img：保存的是叠加在原图上的图，也可以保存过程中其他的自己看看
        print(os.path.join(save_dir, name + str(i) + '.png'))
        i = i + 1

def featuremap_2_heatmap1(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    # heatmap = feature_map[:,0,:,:]*0    #
    heatmap = feature_map[:1, 0, :, :] * 0 #取一张图片,初始化为0
    for c in range(feature_map.shape[1]):   # 按通道
        heatmap+=feature_map[:1,c,:,:]      # 像素值相加[1,H,W]
    heatmap = heatmap.cpu().numpy()    #因为数据原来是在GPU上的
    heatmap = np.mean(heatmap, axis=0) #计算像素点的平均值,会下降一维度[H,W]

    heatmap = np.maximum(heatmap, 0)  #返回大于0的数[H,W]
    heatmap /= np.max(heatmap)      #/最大值来设置透明度0-1,[H,W]
    #heatmaps.append(heatmap)

    return heatmap

def feature_map_channel(features,img_path,save_dir = 'work_dirs/feature_map',name = 'noresbnsie2ltft_'):
    # 随便定义a,b,c,d去取对应的特征层，把通道数变换到最后一个维度，将计算的环境剥离由GPU变成CPU，tensor变为numpy
    a = torch.squeeze(features[0][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    b = torch.squeeze(features[1][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    c = torch.squeeze(features[2][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    d = torch.squeeze(features[3][:1, :, :, :], dim=0).permute(1, 2, 0).detach().cpu().numpy()
    img = cv2.imread(img_path)
    for j,x in enumerate([d]):# x.shape[-1]：表示所有通道数，不想可视化这么多，可以自己写对应的数量
        for i in range(x.shape[-1]):
            heatmap = x[:, :, i]
            # heatmap = np.maximum(heatmap, 0) #一个通道应该不用归一化了
            # heatmap /= np.max(heatmap)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            heatmap0 = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式,0-255,heatmap0显示红色为关注区域，如果用heatmap则蓝色是关注区域
            heatmap = cv2.applyColorMap(heatmap0, cv2.COLORMAP_JET)
            superimposed_img = heatmap * 0.4 + img  # 将热力图应用于原始图像
            # plt.figure()  # 展示
            # plt.title(str(j))
            # plt.imshow(heatmap0) #, cmap='gray'
            # # plt.savefig(os.path.join(save_dir,  name+str(j)+str(i) + '.png'))
            # plt.close()
            cv2.imwrite(os.path.join(save_dir, name + str(j)+str(i) + '.png'), superimposed_img)

