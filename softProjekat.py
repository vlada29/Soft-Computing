# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:29:44 2018

@author: marco
"""

from __future__ import print_function
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

from sklearn.cluster import DBSCAN

import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def hist(image):
    height, width = image.shape[0:2]
    x = range(0, 256)
    y = np.zeros(256)
    
    for i in range(0, height):
        for j in range(0, width):
            pixel = image[i, j]
            y[pixel] += 1
    
    return (x, y)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def winner(output): # output je vektor sa izlaza neuronske mreze
    '''pronaći i vratiti indeks neurona koji je najviše pobuđen'''
    return max(enumerate(output), key=lambda x: x[1])[0]
def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result
def dilate1(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode1(image):
    kernel = np.ones((2,1)) # strukturni element 2x1 blok
    return cv2.erode(image, kernel, iterations=1)
def erode2(image):
    kernel = np.ones((1, 3)) # strukturni element 1x1 blok
    return cv2.erode(image, kernel, iterations=1)
#Funkcionalnost implementirana u OCR basic
def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 30x30'''
    return cv2.resize(region,(30, 30), interpolation = cv2.INTER_NEAREST)
def scale_to_range(image):
    return image / 255
def matrix_to_vector(image):
    return image.flatten()
def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann
def convert_output(outputs):
    return np.eye(len(outputs))

def select_roi(image_orig, image_bin):
 
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    regions_array = []
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        if (w>2 and w<30 and h>10 and h<40):
            region = image_bin[y:y+h,x:x+w];
            regions_array.append([resize_region(region), (x,y,w,h)])
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(255,0,0),1)

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles)-1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index+1]
        distance = next_rect[0] - (current[0]+current[2]) #X_next - (X_current + W_current)
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances


def create_ann():
    ann = Sequential()
    # Postaviti slojeve neurona mreže 'ann'
    ann.add(Dense(244, input_dim=900, activation='sigmoid'))
    ann.add(Dense(34, activation='sigmoid'))
    return ann
    
def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
   
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze


    ann.fit(X_train, y_train, epochs=500, batch_size=1, verbose = 0, shuffle=False) 
      
    return ann
#obican otsu metod koji nije bas radio

def izdvoji(image, param):


    res, thresh= cv2.threshold(image,param,255,0)
    
    selected_regions, numbers, region_distances = select_roi(image.copy(), thresh)
    
    
    return numbers

def izdvojip(image_color, param):

    i_gs = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)
  
    res, thresh= cv2.threshold(i_gs,param,255,0)
    
    selected_regions, numbers, region_distances = select_roi(image_color.copy(), thresh)

    
    return numbers
def obradaSlike(path, mod):
    #dimenzije slike nakon resize
    resized_width = 800
    resized_height = 500
    
    #ucitavanje slike
    imgColor = cv2.imread(path)
    
    #skaliranje slike
    resized = cv2.resize(imgColor,(resized_width,resized_height))
    
    #pretvaranje slike u grayscale
    gray = cv2.cvtColor(resized,cv2.COLOR_RGB2GRAY)

    #iscrtavanje histograma na osnovu grayscale slike
    x,y = hist(gray)
    if mod==1:
        plt.plot(x, y, 'b')
        plt.show()
    
    #pretvaranje slike u hue sat val
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    
    #prebrojavanje piksela koji imaju osvetljenost unutar odredjene granice
    cl1 = 0
    for i in range(35,65):
        cl1=cl1+y[i]
    
    cl2 = 0
    for i in range(65,120):
        cl2=cl2+y[i]
        
    cl3 = 0
    for i in range(120,256):
        cl3=cl3+y[i]
    
    #kreiranje filtera na osnovu klase kojoj slika pripada
    if cl1>cl2 and cl1>cl3:
        if mod==1:
            print('class1')
        lower = np.array([0,0,0])
        upper = np.array([255,255,255])
    elif cl2>cl1 and cl2>cl3:
        if mod==1:
            print('class2')
        lower = np.array([0,50,30])
        upper = np.array([30,255,230])
    else:
        if mod==1:
            print('class3')
        lower = np.array([0,100,30])
        upper = np.array([30,255,255])
    
    
    #kreiranje maske na osnovu filtera
    mask = cv2.inRange(hsv, lower, upper)
    
    #primenjivanje maske na sliku
    res = cv2.bitwise_and(resized,resized, mask = mask)
    
    #erozija pa dilatacija da bi se uklonilo sto vise suma sa slike
    eroded = erode(res)
    final = dilate(eroded)
    
    if mod==1:
        cv2.imshow('Skalirana slika', resized)
    
        cv2.imshow('Posle primene filtera za boju', final)
    
    #pretvaranje finalnog izdvajanja u grayscale
    gray = cv2.cvtColor(final,cv2.COLOR_RGB2GRAY)
    
    #trazenje kontura pomocu Canny algoritma
    edged = auto_canny(gray)
    
    #cv2.imshow('Konture sa filtirane',edged)
    
    #prikupljanje kontura sa slike
    img, contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    #sortiranje kontura,ostavljamo samo 10 najvecih
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    
    #iscrtavanje kontura
    cv2.drawContours(final, contours, -1, (0, 0, 255), 1)
    
    if mod==1:
        cv2.imshow('img',final)
    
    #pronalazenje sirine,visine i x,y koordinata levog gornjeg ugla od najvece konture
    mw = 0
    mh = 0
    mx = 0
    my = 0
    for c in contours:
         x,y,w,h = cv2.boundingRect(c)
         if w>mw :
             if h>mh:
                 mw=w
                 mh=h
                 mx=x
                 my=y
    
    #postavljanje granica za isecanje
    if (my - 0.9*mw) >0:
        topy = int(my - 0.9*mw)
    else:
        topy = 0
    boty = int(my + mh)
    leftx = int(mx)
    rightx = int(mx + mw)
    
    #isecanje autobusa sa slike
    cropped = resized[topy:boty, leftx:rightx]
    
    if mod==1:
        cv2.imshow('Isecen autobus',cropped)
    
    #postavljanje granica za filtriranje na osnovu boje natpisa
    lowerCrop = np.array([60,60,0])
    upperCrop = np.array([120,255,255])
    
    #prebacivanje u hsv
    hsvCrop = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    
    #kreiranje maske za natpis
    maskCrop = cv2.inRange(hsvCrop, lowerCrop, upperCrop)
    
    #primenjivanje maske na sliku
    resCrop = cv2.bitwise_and(cropped,cropped, mask = maskCrop)
    
    if mod==1:
        cv2.imshow('Zuti filter',resCrop)
    
    #da li treba mozda erozija ili dilatacija? nema nekih velikih razlika sa i bez tih filtera
    erodedCrop = resCrop.copy()
    onlyLetters = resCrop.copy()
    
    #prebacivanje u grayscale
    grayCrop = cv2.cvtColor(erodedCrop,cv2.COLOR_RGB2GRAY)
    
    #odredjivanje kontura uz pomoc Canny 
    edgeCropG = auto_canny(grayCrop)
    
    if mod==1:
        cv2.imshow('Konture isecene i filtirane slike po boji',edgeCropG)
    
    #pronalazenje kontura sa te slike
    img2, contours2, hierarchy2 = cv2.findContours(edgeCropG, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    #dimenzije isecene slike
    height,width,chanels = erodedCrop.shape
    
    #izdvajanje svega sto prepozna kao slova u posebnu listu
    letters = []
    for c in contours2:
         cx,cy,cw,ch = cv2.boundingRect(c)
         if(ch>15) and (ch<45):
             letters.append(c)
    
    #iscrtavanje svih kontura koje je prepoznao kao slova na slici
    cv2.drawContours(erodedCrop, letters, -1, (0, 0, 255), 1)
    
    if mod==1:
        cv2.imshow('slova pre filtera',erodedCrop)
    
    #popunjavanje lista za donje i gornje granice svih kontura koje je prepoznao kao slova
    topys=[]
    botys=[]
    for c in letters:
        cx,cy,cw,ch = cv2.boundingRect(c)
        topys.append(cy)
        botys.append(cy+ch)
    
    #prebacivanje gornjih i donjih granica u jednu matricu n x 2
    combined = np.vstack((topys, botys)).T
    
    #DBScan algoritam za grupisanje kontura, minimum kontura za grupu = 3,
    #maksimalno rastojanje izmedju tacaka = 8
    db = DBSCAN(eps=8, min_samples=3).fit(combined)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # Broj klastera koje je DBScan pronasao
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    
    #####################################################Iscrtavanje DBScan dijagrama
    
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        xy = combined[class_member_mask & core_samples_mask]
        if mod==1:
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=8)
    
        xy = combined[class_member_mask & ~core_samples_mask]
        if mod==1:
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)
        
    
    
    if mod==1:
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()

    #################################################################################
    
    #Pravljenje recnika za lakse rukovanje klasterima koje je DBScan vratio,
    cluster_dict = {i: combined[db.labels_==i] for i in range(n_clusters_)}
    
    #Pronalazenje indeksa grupe sa najvise elemenata, posto nam ona sadrzi konture koje predstavljaju slova
    maxInd = 0
    maxNum = 0
    for i in range(0,len(cluster_dict)):
        if len(cluster_dict[i]) > maxNum:
            maxNum=len(cluster_dict[i])
            maxInd=i
    
    #pronalazenje pravougaonika oko kontura koje predstavljaju slova
    topcy = height
    botcy = 0
    lefcx = width
    rigcx = 0
    for c in letters:
         cx,cy,cw,ch = cv2.boundingRect(c)
         if([cy,(cy+ch)] in cluster_dict[maxInd]):
             if cx<lefcx:
                 lefcx=cx
             if cy<topcy:
                 topcy=cy
             if ((cy+ch)>botcy) and ((cy+ch)<height):
                 botcy=(cy+ch)
             if ((cx+cw)>rigcx) and ((cx+cw)<width):
                 rigcx=(cx+cw)
    
    #isecanje pravougaonika oko slova i prikazivanje
    natpis=cropped[topcy:botcy,lefcx:rigcx]
    if mod==1:
        cv2.imshow('natpis',natpis)
        cv2.waitKey()
    
    
    
    
    return natpis

######## olja
def treniraj():  
    width=200
    height=35
    #ucitavanje slika za obucavanje
    image_color1 = load_image('n/2.jpg') #broj 1
    image_color1_resized = cv2.resize(image_color1, (width, height))
    preparedInputs1 = prepare_for_ann(izdvojip(image_color1_resized, 50))
    
    image_color2 = load_image('n/9.jpg') #broj 2 i novonaselje
    image_color2_resized = cv2.resize(image_color2, (width, height))
    preparedInputs2 = prepare_for_ann(izdvojip(image_color2_resized, 40))
    
    image_color3 = load_image('n/23.jpg') #broj 8
    image_color3_resized = cv2.resize(image_color3, (width, height))
    preparedInputs3 = prepare_for_ann(izdvojip(image_color3_resized, 85))
    
    image_color4 = load_image('n/18.jpg') #broj 7 i l
    image_color4_resized = cv2.resize(image_color4, (width, height))
    preparedInputs4 = prepare_for_ann(izdvojip(image_color4_resized, 65))
    
    image_color5 = load_image('n/45.jpg') #p r b
    image_color5_resized = cv2.resize(image_color5, (width, height))
    preparedInputs5 = prepare_for_ann(izdvojip(image_color5_resized, 38))
    
    image_color6 = load_image('n/10.jpg') #broj 9
    image_color6_resized = cv2.resize(image_color6, (width, height))
    preparedInputs6 = prepare_for_ann(izdvojip(image_color6_resized, 65))
    
    image_color7 = load_image('n/41.jpg') #b u k o v a c
    image_color7_resized = cv2.resize(image_color7, (width, height))
    image_color7_resizede=erode1(image_color7_resized)
    preparedInputs7 = prepare_for_ann(izdvojip(image_color7_resizede, 120))
    
    image_color8 = load_image('n/54.jpg') #j g i d
    image_color8_resized = cv2.resize(image_color8, (width, height))
    preparedInputs8 = prepare_for_ann(izdvojip(image_color8_resized, 120))
    
    image_color9 = load_image('n/1.jpg') #5
    image_color9_resized = cv2.resize(image_color9, (width, height))
    preparedInputs9 = prepare_for_ann(izdvojip(image_color9_resized, 120))
    
    image_color10 = load_image('n/11.jpg') #3
    image_color10_resized = cv2.resize(image_color10, (width, height))
    preparedInputs10 = prepare_for_ann(izdvojip(image_color10_resized, 55))
    
        #input definitions: 
    alphabet = ['1',
                '2', 'n', 'o', 'v', 'o', 'n', 'a', 's', 'e', 'lj', 'e',
                '8',
                '7', 'l',
                'p', 'r', 'b',
                '9', 
                '6', '4', 'b', 'u', 'k', 'o', 'v', 'a', 'c',
                'j', 'g', 'i', 'd', '5', '3'
                
                ]
        #prepared input
    pp1=[]
    pp1.append(preparedInputs1[0])
    
    pp2=[]
    pp2=preparedInputs2
    
    pp3=[]
    pp3.append(preparedInputs3[0])
    
    pp4=[]
    pp4.append(preparedInputs4[0])
    pp4.append(preparedInputs4[2])
    
    
    pp5=[]
    pp5.append(preparedInputs5[0])
    pp5.append(preparedInputs5[1])
    pp5.append(preparedInputs5[3])
    
    pp6=[]
    pp6.append(preparedInputs6[0])
    
    pp7=[]
    pp7=preparedInputs7
    
    
    pp8=[]
    pp8.append(preparedInputs8[0])
    pp8.append(preparedInputs8[1])
    pp8.append(preparedInputs8[7])
    pp8.append(preparedInputs8[10])
    
    pp9=[]
    pp9.append(preparedInputs9[0])
    
    pp10=[]
    pp10.append(preparedInputs10[0])
    
    inputs=[]
    inputs=pp1+pp2+pp3+pp4+pp5+pp6+pp7+pp8+pp9+pp10
    
    outputs = convert_output(alphabet)
    
    ann = create_ann()
    ann = train_ann(ann, inputs, outputs)
    return ann
    

def prepoznajSlova(ann, img, mod):
    width=200
    height=35
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # global thresholding
    ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    if mod==1:
        print('Izracunata vrednost pomocu Otsu-a: '+str(ret2))
    
    # plot all the images and their histograms
    
    images = [img, 0, th1,
              img, 0, th2]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
     'Original Noisy Image','Histogram',"Otsu's Thresholding ("+str(ret2)+")"]
    if mod==1:
        for i in range(2):
            plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
            plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
            plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(), 127)
            plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
            plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
            plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
        
        plt.show()
    
    x,y = hist(img)
    
    image_color = cv2.resize(img, (width, height))
    
    
    #prebrojavanje piksela koji imaju osvetljenost unutar odredjene granice
    cl1 = 0
    for i in range(35,100):
        cl1=cl1+y[i]

    cl2 = 0
    for i in range(100,256):
        cl2=cl2+y[i]
        
    otsuv=ret2
        
    if cl1>cl2:
        if mod==1:
            print('class1natpis') 
        ret2=(otsuv+127-30)/2
    else:
        if mod==1:
            print('class2natpis')
        ret2=(otsuv+127+10)/2
       
    if (abs(cl2-cl1)>min(cl1, cl2)*4 and cl2>1000):
        if mod==1:
            print('class0natpisvelika') #jako velika granica
        ret2=otsuv+40
    if (abs(cl2-cl1)>min(cl1, cl2)*4 and cl2<1000):
        if mod==1:
            print('class0natpismala') #jako mala granica
        ret2=otsuv
    if (abs(cl2-cl1)<350):
        if mod==1:
            print('veliki th')
        ret2=otsuv+50
    
    if mod==1:
        print(cl1)
        print(cl2)
    
        

    
    
    #input definitions: 
    alphabet = ['1',
                '2', 'n', 'o', 'v', 'o', 'n', 'a', 's', 'e', 'lj', 'e',
                '8',
                '7', 'l',
                'p', 'r', 'b',
                '9', 
                '6', '4', 'b', 'u', 'k', 'o', 'v', 'a', 'c',
                'j', 'g', 'i', 'd', '5', '3'
                
                ]
    
    

    
    preparedInputsp = prepare_for_ann(izdvoji(image_color, ret2))
    
    
    #############
    #prediction
    
    result = ann.predict(np.array(preparedInputsp, np.float32))
    if mod==1:
        print(display_result(result, alphabet))
    stringovi = ''.join(display_result(result, alphabet))

    return stringovi

memo = {}

def levenshtein(s, t):
    if s == "":
        return len(t)
    if t == "":
        return len(s)
    cost = 0 if s[-1] == t[-1] else 1
       
    i1 = (s[:-1], t)
    if not i1 in memo:
        memo[i1] = levenshtein(*i1)
    i2 = (s, t[:-1])
    if not i2 in memo:
        memo[i2] = levenshtein(*i2)
    i3 = (s[:-1], t[:-1])
    if not i3 in memo:
        memo[i3] = levenshtein(*i3)
    res = min([memo[i1]+1, memo[i2]+1, memo[i3]+cost])
    
    return res

def prepoznajLiniju(inputString):
        #moguce vrednosti autobusa
    busesNS=['1klisa', '1centar', '1liman1',
         '2centar', '2novonaselje', 
         '3petrovaradin', '3centar', '3detelinara',
         '7anovonaselje',
         '8novonaselje', '8centar',
         '9novonaselje', '9petrovaradin',
         '12centar',
         '43novisad',
         '71bocke',
         'zagarazu',
         '62novisad',
         '64novisad']

    daljine={} # daljina i string u busesNS
    for i in range(len(busesNS)):
        daljine[busesNS[i]]=levenshtein(inputString, busesNS[i])

    minDaljina=min(daljine, key=daljine.get)
    #print(min(daljine, key=daljine.get)) # najmanja daljjina, string koji najvise lici na prepoznati

    linije = {"1klisa": "1 KLISA-CENTAR-LIMAN I",
            "1centar": "1 KLISA-CENTAR-LIMAN I",
            "1liman1": "1 KLISA-CENTAR-LIMAN I",
            "2centar": "2 CENTAR-NOVO NASELJE",
            "2novonaselje": "2 CENTAR-NOVO NASELJE",
            "3petrovaradin": "3 PETROVARADIN-CENTAR-DETELINARA",
            "3centar": "3 PETROVARADIN-CENTAR-DETELINARA",
            "3detelinara": "3 PETROVARADIN-CENTAR-DETELINARA",
            "7anovonaselje": "7A NOVO NASELJE-Ž. STANICA-LIMAN",
            "8novonaselje": "8 NOVO NASELJE-CENTAR-LIMAN",
            "9novonaselje": "9 NOVO NASELJE-LIMAN-PETROVARADIN",
            "12centar": "12 CENTAR-TELEP",
            "43novisad": "43 NOVI SAD",
            "62novisad": "62 NOVI SAD",
            "64novisad": "64 NOVI SAD",
            "71bocke": "71 BOCKE",
            "zagarazu":"ZA GARAŽU"}

# izabrana linija
    return linije[minDaljina]


def PrepoznavanjeAutobuskihLinija(ann, folder, number):
    procenat = 'Procenat uspešnosti prepoznavanja na osnovu unetih slika je: '
    tacno = 0.0
    #ucitavanje slika iz foldera
    with open('busesTest.txt') as f:
        content = f.readlines()

    content = [x.strip() for x in content] 
    for i in range(number):
        slova=prepoznajSlova(ann, obradaSlike(str(folder)+'/'+str(i+1)+'.jpg', 2), 2)
        linija=prepoznajLiniju(slova)
        print('Autobuska linija koja je na slici je: '+str(linija))
        if (linija==content[i]):
            tacno=tacno+1.0
    
    procenat=procenat+str((tacno/number)*100.0)+'%.'
        
    return procenat

#neuronska
nmreza=treniraj()


#detaljno prepoznavanje za 1 sliku
#slova=prepoznajSlova(nmreza, obradaSlike('buses/6.jpg', 1), 1)
#print('Autobuska linija koja je na slici je: '+str(prepoznajLiniju(slova)))
    

#prepoznavanje test primera i racunanje preciznosti
print(PrepoznavanjeAutobuskihLinija(nmreza, 'buses', 10))


