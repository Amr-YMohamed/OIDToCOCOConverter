import pandas as pd
import numpy as np
import datetime
import argparse
import json


def getReqCols (ann_path, description_path, rotation_path, sizes_path, chunksize):

    #reading classes in chunks
    for chunk in pd.read_csv(description_path, chunksize=chunksize, header=None):
        print("loading bunch of classes for processing based on chunkSize...")
        classLabels = chunk.iloc[:,0].to_numpy() #classcodes
        
        print("fetching annotations from its file...")
        #column names of concern from annotation file
        cols = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", 
        "IsGroupOf", "IsDepiction", "IsInside"]
        #reading the dataframe that corresponds to the classes, cols of concern
        annFile = pd.read_csv(ann_path, usecols = cols)
        labelNames = annFile.LabelName.to_numpy()
        boolCond = pd.Series(np.in1d(labelNames, classLabels))
        annFile = annFile[boolCond.values]
        chunk_imgs = annFile.ImageID.to_numpy()

        print("fetching rotations from its file...")
        #column names of concern from rotation file
        cols = ["ImageID", "OriginalURL", "License"]
        #reading the dataframe that corresponds to the classes, cols of concern
        rotations = pd.read_csv(rotation_path, usecols = cols)
        img_ids = rotations.ImageID.to_numpy()
        boolCond = pd.Series(np.in1d(img_ids, chunk_imgs)) #bottleneck needs improving
        rotations = rotations[boolCond.values]   
        
        print("fetching sizes from its file...")
        #column names of concern from rotation file
        cols = ["image_id", "image_w", "image_h"]
        #reading the dataframe that corresponds to the classes, cols of concern
        sizes = pd.read_csv(sizes_path, usecols = cols)
        img_ids = sizes.image_id.to_numpy()
        boolCond = pd.Series(np.in1d(img_ids, chunk_imgs)) #bottleneck needs improving
        sizes = sizes[boolCond.values]   

        yield annFile, chunk, sizes, rotations, sizes

def getclassCols (ann_path, description_path, rotation_path, sizes_path, className):

    chunk = pd.read_csv(description_path, header=None)
    chunk = chunk.loc[(chunk.iloc[:,1]).str.lower() == className]
    print(chunk)

    print("loading bunch of classes for processing based on chunkSize...")
    classLabels = chunk.iloc[:,0].to_numpy() #classcodes
    
    print("fetching annotations from its file...")
    #column names of concern from annotation file
    cols = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsOccluded", "IsTruncated", 
    "IsGroupOf", "IsDepiction", "IsInside"]
    #reading the dataframe that corresponds to the classes, cols of concern
    annFile = pd.read_csv(ann_path, usecols = cols)
    labelNames = annFile.LabelName.to_numpy()
    boolCond = pd.Series(np.in1d(labelNames, classLabels))
    annFile = annFile[boolCond.values]
    chunk_imgs = annFile.ImageID.to_numpy()

    print("fetching rotations from its file...")
    #column names of concern from rotation file
    cols = ["ImageID", "OriginalURL", "License"]
    #reading the dataframe that corresponds to the classes, cols of concern
    rotations = pd.read_csv(rotation_path, usecols = cols)
    img_ids = rotations.ImageID.to_numpy()
    boolCond = pd.Series(np.in1d(img_ids, chunk_imgs)) #bottleneck needs improving
    rotations = rotations[boolCond.values]   
    
    print("fetching sizes from its file...")
    #column names of concern from rotation file
    cols = ["image_id", "image_w", "image_h"]
    #reading the dataframe that corresponds to the classes, cols of concern
    sizes = pd.read_csv(sizes_path, usecols = cols)
    img_ids = sizes.image_id.to_numpy()
    boolCond = pd.Series(np.in1d(img_ids, chunk_imgs)) #bottleneck needs improving
    sizes = sizes[boolCond.values]   

    yield annFile, chunk, sizes, rotations, sizes

def addClassDesc(desc):
    classes = []
    indicies = pd.Series(range(len(desc.index)))
    #uncomment next line if the original file of class desc will be needed
    #for index, code, name in zip(desc.index, desc.iloc[:,0], desc.iloc[:,1]): 
    for index, code, name in zip(indicies, desc.iloc[:,0], desc.iloc[:,1]):
        clas = {}
        clas['id'] = index + 1 #number in the file (i.e. row number)
        clas['name'] = name #class name(person, dog, watch,...)
        clas['freebase_id'] = code #class id(/m/01g317,...)
        classes.append(clas)
    return classes

def addImageSizes(rotations, sizes):
    images = []
    sizes = dict(zip(sizes.image_id, zip(sizes.image_w, sizes.image_h)))

    #iterate over all images in this chunk
    for id, url, lic in zip(rotations.ImageID, rotations.OriginalURL, rotations.License):
        img = {}
        img['id'] = id
        img['file_name'] = id + '.jpg'
        img['license'] = lic
        img['url'] = url
        img['width'] = int(sizes[id][0])
        img['height'] = int(sizes[id][1])

        # Add to list of images
        images.append(img)
        
    return images

def addAnnotations(annotations, imageData, classMapping):
    returned_annotations = []
    indicies = pd.Series(range(len(annotations.index)))

    imgs = {img['id']: img for img in imageData}
    classMapping = {clas['freebase_id']: clas for clas in classMapping}

    #num_instances = len(original_annotations_dict)
    for index, imageID, labelName, XMin, XMax, YMin, YMax, IsOccluded, IsTruncated, IsGroupOf, IsDepiction, IsInside in zip(
        indicies, annotations.ImageID, annotations.LabelName, annotations.XMin, annotations.XMax, annotations.YMin, 
        annotations.YMax, annotations.IsOccluded, annotations.IsTruncated, annotations.IsGroupOf, annotations.IsDepiction, 
        annotations.IsInside):
        # set individual instance id
        # use start_index to separate indices between dataset splits
        ann = {}
        ann['id'] = index
        ann['image_id'] = imageID

        ann['freebase_id'] = labelName
        ann['category_id'] = classMapping[labelName]['id']
        ann['iscrowd'] = False
        
        xmin = float(XMin) * imgs[imageID]['width']
        ymin = float(YMin) * imgs[imageID]['height']
        xmax = float(XMax) * imgs[imageID]['width']
        ymax = float(YMax) * imgs[imageID]['height']
        dx = xmax - xmin
        dy = ymax - ymin
        ann['bbox'] = [round(a, 2) for a in [xmin , ymin, dx, dy]]
        ann['area'] = round(dx * dy, 2)
        ann['isoccluded'] = IsOccluded
        ann['istruncated'] = IsTruncated
        ann['isgroupof'] = IsGroupOf
        ann['isdepiction'] = IsDepiction
        ann['isinside'] = IsInside

        returned_annotations.append(ann)   
    return returned_annotations

def converOID(ann_path, description_path, rotation_path, sizes_path, chunksize):
    oi = {}
    writeCommon(oi)
    ind = 1
    for ann, desc, sizes, rotations, sizes in getReqCols (ann_path, description_path, rotation_path, sizes_path, chunksize):

        # Convert category information
        print('converting category info')
        oi['categories'] = addClassDesc(desc)

        # Convert image mnetadata
        print('converting image info ...')
        oi['images'] = addImageSizes(rotations, sizes)


        # Convert annotations
        print('converting annotations ...')
        oi['annotations'] = addAnnotations(ann, oi['images'], oi['categories'])

        filename = str(ind) + ".json"
        print('writing output to {}'.format(filename))
        json.dump(oi,  open(filename, "w"))
        ind = ind + 1
    
def convertSingleClass(ann_path, description_path, rotation_path, sizes_path, className):

    oi = {}
    writeCommon(oi)
    ann, desc, sizes, rotations, sizes  = getclassCols (ann_path, description_path, rotation_path, sizes_path, className)

    # Convert category information
    print('converting category info')
    oi['categories'] = addClassDesc(desc)

    # Convert image mnetadata
    print('converting image info ...')
    oi['images'] = addImageSizes(rotations, sizes)


    # Convert annotations
    print('converting annotations ...')
    oi['annotations'] = addAnnotations(ann, oi['images'], oi['categories'])

    filename = str(className) + ".json"
    print('writing output to {}'.format(filename))
    json.dump(oi,  open(filename, "w"))


def writeCommon(oi):
    print('loading original annotations ... Done')

    # Add basic dataset info
    print('adding basic dataset info')
    oi['info'] = {'contributos': 'Vittorio Ferrari, Tom Duerig, Victor Gomes, Ivan Krasin,\
                  David Cai, Neil Alldrin, Ivan Krasinm, Shahab Kamali, Zheyun Feng,\
                  Anurag Batra, Alok Gunjan, Hassan Rom, Alina Kuznetsova, Jasper Uijlings,\
                  Stefan Popov, Matteo Malloci, Sami Abu-El-Haija, Rodrigo Benenson,\
                  Jordi Pont-Tuset, Chen Sun, Kevin Murphy, Jake Walker, Andreas Veit,\
                  Serge Belongie, Abhinav Gupta, Dhyanesh Narayanan, Gal Chechik',
                  'description': 'Open Images Dataset V6',
                  'url': 'https://storage.googleapis.com/openimages/web/index.html',
                  'version': 'V6', 'year': 2020}

    # Add license information
    print('adding basic license info')
    oi['licenses'] = [{'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License', 'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/'},
                      {'id': 2, 'name': 'Attribution-NonCommercial License', 'url': 'http://creativecommons.org/licenses/by-nc/2.0/'},
                      {'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License', 'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/'},
                      {'id': 4, 'name': 'Attribution License', 'url': 'http://creativecommons.org/licenses/by/2.0/'},
                      {'id': 5, 'name': 'Attribution-ShareAlike License', 'url': 'http://creativecommons.org/licenses/by-sa/2.0/'},
                      {'id': 6, 'name': 'Attribution-NoDerivs License', 'url': 'http://creativecommons.org/licenses/by-nd/2.0/'},
                      {'id': 7, 'name': 'No known copyright restrictions', 'url': 'http://flickr.com/commons/usage/'},
                      {'id': 8, 'name': 'United States Government Work', 'url': 'http://www.usa.gov/copyright.shtml'}]

    return oi

if __name__ == '__main__':

    start = datetime.datetime.now()

    parser = argparse.ArgumentParser("script converts OID csv file to coco json format")
    parser.add_argument("--a", help = "The path to the annotation file.", type = str)
    parser.add_argument("--i", help = "The path to image sizes file.", type = str)
    parser.add_argument("--d", help = "The path to class description file.", type = str)
    parser.add_argument("--r", help = "The path to rotation file.", type = str)

    parser.add_argument("--c", help = "class name (if set chunk size will be ignored).", type = str, default = "")
    parser.add_argument("--s", help = "chunk size in each json.", type = int, default = 19)
    args = parser.parse_args()

    ann_path = args.a
    sizes_path = args.i
    description_path = args.d
    rotation_path = args.r
    className = args.c
    chunksize = args.s

    if className != "":
        convertSingleClass(ann_path, description_path, rotation_path, sizes_path, className)
    else:
        converOID(ann_path, description_path, rotation_path, sizes_path, chunksize)

    #getReqCols("oidv6-train-annotations-bbox.csv", "class-descriptions-boxable.csv", "")

    end = datetime.datetime.now()
    print("process time: %s" % (end-start))