import os

os.system(
    "wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O VOCtrainval_11-May-2012.tar")
os.system(
    "wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar -O VOCtrainval_06-Nov-2007.tar")
os.system("wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar -O VOCtest_06-Nov-2007.tar")

os.system("wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar -O VOCtest_06-Nov-2007.tar")

os.system("mkdir -p data/VOC/2007_test/")
os.system("mkdir -p data/VOC/2007_trainval/")
os.system("mkdir -p data/VOC/2012_trainval/")
os.system("tar -xvf VOCtest_06-Nov-2007.tar -C data/VOC/2007_test/")
os.system("tar -xvf VOCtrainval_06-Nov-2007.tar -C data/VOC/2007_trainval/")
os.system("tar -xvf VOCtrainval_11-May-2012.tar -C data/VOC/2012_trainval/")
os.system('rm -rf *.tar')
os.system("mv data/VOC/2007_test/VOCdevkit/VOC2007/* data/VOC/2007_test/")
os.system("mv data/VOC/2007_trainval/VOCdevkit/VOC2007/* data/VOC/2007_trainval/")
os.system("mv data/VOC/2012_trainval/VOCdevkit/VOC2012/* data/VOC/2012_trainval")
