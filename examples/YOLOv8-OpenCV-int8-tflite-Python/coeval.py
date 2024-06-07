import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
# import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm','bbox','keypoints']
annType = annType[0]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

dataDir='../'
dataType='val2014'
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
annFile = r'C:\\work\\datasets\\coco\\test-dev\\instances_val2017.json'
cocoGt=COCO(annFile)


resFile=r'C:\Users\velonica\Downloads\seg.json'
# resFile = "instances_val2014_fakebbox100_results.json"
cocoDt=cocoGt.loadRes(resFile)

imgIds=sorted(cocoGt.getImgIds())
imgIds=imgIds[0:100]
imgId = imgIds[np.random.randint(100)]

cocoEval = COCOeval(cocoGt,cocoDt,annType)
#将 cocoEval 对象的 imgIds 参数设置为包含要评估的图像ID的列表 imgIds。
#只评估这些指定图像的预测结果和真实标签，而不是整个数据集。
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()