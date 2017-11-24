
# coding: utf-8

# In[9]:


import sys
sys.path.append('/home/walter040422/Downloads/caffe-master/python')

from caffe import layers as L, params as P, to_proto
root='/home/walter040422/'
deploy=root+'Downloads/caffe-master/my-projects/re_test/deploy.prototxt'     #文件保存路径


def create_deploy():
    #少了第一层数据层
    #第二层，卷积层
    conv1=L.Convolution(bottom='data', kernel_size=11, stride=4, num_output=96, pad=0,
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type='gaussian',std=0.01),
                        bias_filler=dict(type='constant',value=0)                                          
                       )
    #第三层，激活函数层
    relu1=L.ReLU(conv1, in_place=True)
    #第四层，池化层
    pool1=L.Pooling(relu1, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    #第五层，LRN层
    norm1=L.LRN(pool1, local_size=5, alpha=1e-4, beta=0.75)
    #第六层，卷积层
    conv2=L.Convolution(norm1, kernel_size=5, stride=1, num_output=256, pad=2, group=2,
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type='gaussian',std=0.01),
                        bias_filler=dict(type='constant',value=1)                                                                   
                       )
    
    #第七层，激活函数层
    relu2=L.ReLU(conv2, in_place=True)
    #第八层，池化层
    pool2=L.Pooling(relu2, pool=P.Pooling.MAX, kernel_size=3, stride=2)
    #第九层，LRN层
    norm2=L.LRN(pool2, local_size=5, alpha=1e-4, beta=0.75)
    #第十层，卷积层
    conv3=L.Convolution(norm2, kernel_size=3, stride=1, num_output=384, pad=1,
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type='gaussian',std=0.01),
                        bias_filler=dict(type='constant',value=0)                   
                       )
    #第十一层，激活函数层
    relu3=L.ReLU(conv3, in_place=True)
    #第十二层，卷积层
    conv4=L.Convolution(relu3, kernel_size=3, stride=1, num_output=384, pad=1,group=2,
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type='gaussian',std=0.01),
                        bias_filler=dict(type='constant',value=1)                     
                       )

    #第十三层，激活函数层
    relu4=L.ReLU(conv4, in_place=True)
    #第十四层，卷积层
    conv5=L.Convolution(relu4, kernel_size=3, stride=1, num_output=256, pad=1,group=2,
                        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                        weight_filler=dict(type='gaussian',std=0.01),
                        bias_filler=dict(type='constant',value=1)     
                       )
    
    
    #第十五层，激活函数层
    relu5=L.ReLU(conv5, in_place=True)   
    #第十六层，池化层
    pool5=L.Pooling(relu5, pool=P.Pooling.MAX, kernel_size=3, stride=2)    
    #第十七层,全连接层
    fc6 = L.InnerProduct(pool5, num_output=4096,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='gaussian',std=0.005),
                         bias_filler=dict(type='constant',value=1)
                        )
    #第十八层，激活函数层
    relu6=L.ReLU(fc6, in_place=True)
    #第十九层，Dropout层
    drop6=L.Dropout(relu6, dropout_ratio=0.5, in_place=True)
    #第二十层，全连接层
    fc7 = L.InnerProduct(drop6, num_output=4096,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                         weight_filler=dict(type='gaussian',std=0.005),
                         bias_filler=dict(type='constant',value=1)
                        )
    #第二十一层，激活函数层
    relu7=L.ReLU(fc7, in_place=True)
    #第二十二层，Dropout层
    drop7=L.Dropout(relu7, dropout_ratio=0.5)
    #第二十三层，全连接层
    fc8=L.InnerProduct(drop7, num_output=1000, 
                       param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                       weight_filler=dict(type='gaussian',std=0.01),
                       bias_filler=dict(type='constant', value=0)
                      )
    
    #最后没有accuracy层，但有一个Softmax层
    prob=L.Softmax(fc8)
    return to_proto(prob)
    
    

def write_deploy():
    with open(deploy, 'w') as f:
        f.write('name:"Caffenet"\n')
        f.write('input:"data"\n')
        f.write('input_dim:1\n')
        f.write('input_dim:3\n')
        f.write('input_dim:227\n')
        f.write('input_dim:227\n')
        f.write(str(create_deploy()))
        
        
     
if __name__=='__main__':
    write_deploy()


    
    
    

    

