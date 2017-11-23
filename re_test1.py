
# coding: utf-8

# In[ ]:


import sys 
sys.path.append('/home/walter040422/Downloads/caffe-master/python')

import caffe
from caffe import layers as L, params as P, proto, to_proto
#设定文件的保存路径
root='/home/walter040422/'               #根目录
train_lmdb=root+'Downloads/caffe-master/my-projects/re_test/train_lmdb'        #训练数据lmdb文件的位置
val_lmdb=root+'Downloads/caffe-master/my-projects/re_test/val_lmdb'            #测试数据lmdb文件的位置
mean_file=root+'Downloads/caffe-master/my-projects/re_test/mean.binaryproto'   #均值文件的位置
train_proto=root+'Downloads/caffe-master/my-projects/re_test/train.prototxt'   #生成的训练配准文件保存的位置
val_proto=root+'Downloads/caffe-master/my-projects/re_test/val.prototxt'       #生成的测试配置文件保存的位置
solver_proto=root+'Downloads/caffe-master/my-projects/re_test/solver.prototxt'    #生成的参数文件保存的位置


#编写一个函数，生成配置文件prototxt
def CaffeNet(lmdb, batch_size, include_acc=False):
#    n=caffe.NetSpec()
    #第一层，数据输入层，以lmdb格式输入
    data, label = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
                        transform_param=dict(crop_size=227, mean_file=mean_file, mirror=True))
    #第二层，卷积层
    conv1=L.Convolution(data, kernel_size=11, stride=4, num_output=96, pad=0,
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

    #第二十四层，Softmax层
    loss=L.SoftmaxWithLoss(fc8, label)

    
    
    
    if include_acc:               #test阶段需要有accuracy层
        acc=L.Accuracy(fc8, label)
        return to_proto(loss, acc)
    else:
        return to_proto(loss)

    
    
   
def write_net():
    #写入train.prototxt
    with open(train_proto, 'w') as f:
        f.write(str(CaffeNet(train_lmdb, batch_size=256)))
        
    #写入val.prototxt
    with open(val_proto, 'w') as f:
        f.write(str(CaffeNet(val_lmdb, batch_size=50, include_acc=True)))

        
        
        
        
def gen_solver(solver_file, train_net, test_net):
    s=proto.caffe_pb2.SolverParameter()
    s.train_net=train_net
    s.test_net.append(test_net)
    s.test_interval=50            #400/64， 测试间隔参数：训练完一次所有的图片，进行一次测试
    s.test_iter.append(2)         #100/50, 测试迭代次数，需要迭代2次，才完成一次所有数据的测试
    s.base_lr=1e-3                #基础学习率
    s.lr_policy='step'            #学习率变化规则
    s.gamma=0.1                   #学习率变化指数
    s.stepsize=100                #学习率变化频率
    s.display=20                  #屏幕显示间隔
    s.max_iter=500                #10 epochs, 50*10, 最大训练次数
    s.momentum=0.9                #动量    
    s.weight_decay=5e-3           #权值衰减项    
    s.solver_mode=proto.caffe_pb2.SolverParameter.GPU     #加速
    s.snapshot_prefix=root+'Downloads/caffe-master/my-projects/re_test/caffenet'       #caffemodel前缀    
    s.snapshot=500                #保存caffemodel的间隔
    s.type='SGD'                  #优化算法    
    #写入solver.prototxt
    with open(solver_file, 'w') as f:
        f.write(str(s))
        
        
        
#开始训练
def training(solver_proto):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver=caffe.SGDSolver(solver_proto)
    solver.solve()

    
if __name__=='__main__':
    write_net()
    gen_solver(solver_proto,train_proto,val_proto)
    training(solver_proto)

   
    
    
    
    
    






