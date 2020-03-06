__author__ = 'QiYE'
import os
# files = os.listdir('F:/HuaweiProj/data/mega/annotation_after_shift_by_qi_20180113/')
# print()
# for file_name in files:
#     subfolder = file_name.split('_')[0]
#     print(subfolder)
#
#
# # subjects=['Yang']
def get_img_path_from_annotation_file(dataset, base_path,subjects):
    new_txt_file = open("%s/%s_annotation.txt"%(base_path,dataset),mode='w',encoding='utf-8',newline='')
    for subject in subjects:
        subfolder = '%s/%s'%(base_path,subject)
        files = os.listdir(subfolder)

        for file_name in files:

            segfolder = file_name.split('_')[0]
            print('num files',len(files), subject,segfolder)

            with open('%s/%s'%(subfolder,file_name), mode='r',encoding='utf-8',newline='') as f:
                lines = list(f.readlines())

                for line in lines:
                    part = line.split('\t')
                    new_txt_file.write('%s/%s/image_D%s'%(subject,segfolder,part[0]))

                    for i in part[1:]:
                        new_txt_file.write('\t')
                        new_txt_file.write(i)
            f.close()
    new_txt_file.close()

def read_annotation_from_txtfile(base_path,dataset):
    new_txt_file = open("%s/%s_annotation.txt"%(base_path,dataset),mode='r',encoding='utf-8',newline='')
    lines = list(new_txt_file)
    for line in lines:

        part = line.split('\t')
        print(part)

import numpy
def get_num_img(dataset, base_path,subjects):
    num=[]
    for subject in subjects:
        subfolder = '%s/%s'%(base_path,subject)
        files = os.listdir(subfolder)
        num_lines = 0
        for file_name in files:

            segfolder = file_name.split('_')[0]
            # print('num files',len(files), subject,segfolder)

            with open('%s/%s'%(subfolder,file_name), mode='r',encoding='utf-8',newline='') as f:
                num_lines+=len(list(f.readlines()))/5
        num.append(num_lines)
        print(subject,num_lines)
    print(numpy.sum(num[:-2]),numpy.sum(num[:-1]),numpy.sum(num))
if __name__=='__main__':
    subjects = ['Guillermo','Caner','Pamela','Patrick','Qi','sara','seung','ShanxinV7','vassileios','Xinghao']
    # dataset='train'
    # subjects = ['Xinghao']
    dataset='test'
    get_num_img(dataset=dataset,base_path='F:/HuaweiProj/data/mega/annotation_after_shift_by_qi_20180113',subjects=subjects)
    # get_img_path_from_annotation_file(dataset='test',base_path='F:/HuaweiProj/data/mega/annotation_after_shift_by_qi_20180113',subjects=['Yang'])
    # read_annotation_from_txtfile(dataset='test',base_path='F:/HuaweiProj/data/mega/annotation_after_shift_by_qi_20180113')
    #