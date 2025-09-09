#coding=utf-8
#!/usr/bin/python
import os

file_path = "/data/dataset/JCJQ/split_1024_jcjq_ss/test/images_0.1_mask_0.9_3"
# os.listdir(file)会历遍文件夹内的文件并返回一个列表
path_list = os.listdir(file_path)
# print(path_list)
# 定义一个空列表,我不需要path_list中的后缀名
path_name=[]
# 利用循环历遍path_list列表并且利用split去掉后缀名
for i in path_list:
    path_name.append(i)

# 排序一下
#path_name.sort()

for file_name in path_name:
    # "a"表示以不覆盖的形式写入到文件中,当前文件夹如果没有"save.txt"会自动创建
    with open("/data/dataset/JCJQ/split_1024_jcjq_ss/test/images_0.1_mask_0.9_3.txt","a") as f:
        # f.write(str(file_name))
        for i in file_name:
            f.write(str(i))
        f.write('\r\n')
    f.close()
