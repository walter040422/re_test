
# coding: utf-8

# In[ ]:


import sys
sys.path.append('/home/walter040422/Downloads/caffe-master/python')

import caffe
import numpy as np
root='/home/walter040422/'   #根目录
deploy=root+'Downloads/caffe-master/my-projects/re_test/deploy.prototxt'    #deploy文件
caffe_model=root+'Downloads/caffe-master/my-projects/re_test/caffenet_iter_500.caffemodel'    #训练好的caffemodel
img=root+'Documents/re/train/horse/737.jpg'       #随机找的一张待测图片
labels_filename=root + 'Downloads/caffe-master/my-projects/re_test/labels.txt'     #类别名称文件，将数字标签转换回类别名称

net=caffe.Net(deploy, caffe_model, caffe.TEST)   #加载model和network

#图片预处理设置
transformer=caffe.io.Transformer({'data':net.blobs['data'].data.shape})    #设定图片的shape格式（1,3,227,227）
transformer.set_transpose('data',(2,0,1))                                  #改变维度的顺序，由原始图片（227,227,3）变为（3,227,227）
transformer.set_raw_scale('data',255)                                      #缩放到【0,255】之间
transformer.set_channel_swap('data',(2,1,0))                               #交换通道，将图片由RGB变为BGR

im=caffe.io.load_image(img)                     #加载图片
net.blobs['data'].data[...]=transformer.preprocess('data',im)              #执行上面设置的图片预处理操作，并将图片载入到blob中

#执行测试
out=net.forward()

labels=np.loadtxt(labels_filename, str, delimiter='\t')       #读取类别名称文件
prob=net.blobs['Softmax1'].data[0].flatten()                  #取出最后一层（Softmax）属于某个类别的概率值，并打印
print prob
order=prob.argsort()[-1]                                      #将概率值排序，取出最大值所在的符号
print 'the class is:', labels[order]                          #将该序号转换成对应的类别名称，并打印






