'''
Title           :make_predictions_1.py
Description     :This script makes predictions using the 1st trained model and generates a submission file.
Author          :Adil Moujahid
Date Created    :20160623
Date Modified   :20160625
version         :0.2
usage           :python make_predictions_1.py
python_version  :2.7.11
'''

import os,sys
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

if len(sys.argv) < 2:
   print  "No test file"
   sys.exit()

root='/workspace/'
deploy=root+'deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/caffenet_deploy_1.prototxt'
caffe_model=root+'deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/caffe_model_1_iter_40000.caffemodel'
mean_file=root+'deeplearning-cats-dogs-tutorial/input/mean.binaryproto'
labels_filename=root+'deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/labels1.txt'
csv_filename=root+'deeplearning-cats-dogs-tutorial/caffe_models/caffe_model_1/submission_model_1.csv'

dir=sys.argv[1]
filelist=[]
filenames=os.listdir(dir)
for fn in filenames:
    fullfilename=os.path.join(dir,fn)
    filelist.append(fullfilename)

caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

'''
Image processing helper function
'''

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights 
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open(mean_file) as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
net = caffe.Net(deploy,caffe_model,caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

'''
Making predicitions
'''
#Reading image paths
#test_img_paths = [img_path for img_path in glob.glob('/workspace/cc/catdog_test/*jpg')]

#Making predictions
test_ids = []
preds = []
for img_path in filelist :
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']

    prob=net.blobs['prob'].data[0].flatten()
    labels=np.loadtxt(labels_filename,str,delimiter='/t')
    index1=prob.argsort()[-1]
    index2=prob.argsort()[-2]

    test_ids = test_ids + [img_path.split('/')[-1][:-4]]
    preds = preds + [pred_probas.argmax()]

    print img_path
    print labels[index1],pred_probas.argmax()
    print labels[index1],'--',prob[index1]
    print labels[index2],'--',prob[index2]
    print '-------'

'''
Making submission file
'''
with open(csv_filename,"w") as f:
    f.write("id,label\n")
    for i in range(len(test_ids)):
        f.write(str(test_ids[i])+","+str(preds[i])+"\n")
f.close()
