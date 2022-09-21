'''Author Mizanu Zelalem mizanu143@gmail.com
   BCCDMS
   2021
'''
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from .models import *
import base64
import os
import io
import PIL.Image as Image
from keras.applications.resnet import ResNet50
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.applications.resnet import ResNet152
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, MaxPooling2D, Convolution2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import GaussianNoise
import tensorflow as tf

import cv2
import os

import numpy as np
import keras
from keras.layers import Dense, Dropout, Input, Flatten, Conv2D, MaxPooling2D, Concatenate
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split


def deepModel():
    model = ResNet50(include_top=False, input_shape=(300, 300, 3))
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(64, activation='relu')(flat1)
    output = Dense(4, activation='softmax')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def deepModel_with_gas():
    from keras.layers import GaussianNoise
    model = Sequential()
    model.add(Conv2D(64, 3, padding="same",
              activation="relu", input_shape=(100, 100, 3)))
    model.add(MaxPool2D(2, 2))

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(64, 3, padding="same", activation="relu"))

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu',
              bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(GaussianNoise(0.01))
    model.add(Dense(64, activation="relu",
              bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(Dense(3, activation="softmax"))
    opt = Adam(lr=1e-5)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True), metrics=['accuracy'])
    return model


def cervixTypeModel1():
    from keras.layers import GaussianNoise
    model = Sequential()
    model.add(Conv2D(64, 3, padding="same",
              activation="relu", input_shape=(100, 100, 3)))
    model.add(MaxPool2D(2, 2))

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(64, 3, padding="same", activation="relu"))

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu',
              bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(GaussianNoise(0.01))
    model.add(Dense(64, activation="relu",
              bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(Dense(3, activation="softmax"))
    opt = Adam(lr=1e-5)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True), metrics=['accuracy'])
    return model


def cervixTypeClassifier():
    # round 4
    model = Sequential()
    # use input_shape=(3, 64, 64)
    model.add(Convolution2D(32, 3, 3, activation='relu',
              input_shape=(250, 250, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(36, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adamax',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def pap_smear_classifier():
    model = Sequential()
    model.add(Conv2D(4, (3, 3), strides=(2, 2), padding='same',
              input_shape=(224, 224, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))

    model.add(Flatten())
    model.add(Dense(units=14, activation='relu'))
    model.add(Dense(units=2, activation='sigmoid'))

    adam = Adam(lr=1e-5)
    model.compile(optimizer='adamax', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def pap_smear_classifier_2():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same',
              input_shape=(224, 224, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=12, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))

    adam = Adam(lr=1e-5)
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def breast_cancer_binary_classifier_model1():
    model = Sequential()
    model.add(Conv2D(64, 3, padding="same",
              activation="relu", input_shape=(224, 224, 3)))
    model.add(MaxPool2D(2, 2))

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(64, 3, padding="same", activation="relu"))

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu',
              bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(GaussianNoise(0.01))
    model.add(Dense(64, activation="relu",
              bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(Dense(2, activation="softmax"))
    opt = Adam(lr=1e-5)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def setsession(request, username):
    request.session['bccdms_user'] = username
    request.session.modified = True


def getsession(request):
    username = request.session.get('bccdms_user')
    return username


def setPatientSession(request, id):
    request.session['patient_id'] = id
    request.session.modified = True


def getPatientSession(request):
    id = request.session.get('patient_id')
    return id


def logout(request):
    del request.session['bccdms_user']
    request.session.modified = True
    return redirect(getStarted)


def getStarted(request):
    return render(request, "acdms_get_started.html", {})


def login(request):
    try:
        username = request.POST.get("username")
        password = request.POST.get("password")
        print(username)
        userObj = pathologist_account.objects.filter(
            username=username, password=password)
        if len(userObj) > 0:
            userObj = userObj[0]
            setsession(request, userObj.username)
            return redirect(Home)
        else:
            return render(request, "acdms_get_started.html", {"ErrorMessage": "Incorrect username and or password"})
    except Exception as e:
        print(e)
        return redirect(getStarted)


def Home(request):
    username = getsession(request)
    userObj = pathologist_account.objects.filter(username=username)
    if len(userObj) > 0:
        fullName = userObj[0].fullName
        return render(request, "acdms_home.html", {"fullName": fullName})
    else:
        return redirect(getStarted)


def addNewRecord(request):
    username = getsession(request)
    userObj = pathologist_account.objects.filter(username=username)
    if len(userObj) > 0:
        fullName = userObj[0].fullName
        return render(request, "add_new_record.html", {"fullName": fullName})
    else:
        return redirect(getStarted)


def savePetientInfo(request):
    try:
        cardNo = request.POST.get("cardNo")
        fullName = request.POST.get("fullName")
        address = request.POST.get("address")
        phone = request.POST.get("phone")
        dateOfBirth = request.POST.get("dateOfBirth")
        patient_infoObject = patient_info(patient_id=cardNo, patient_fullName=fullName,
                                          patient_address=address, patient_phone=phone, patient_dateOfBirth=dateOfBirth)
        patient_infoObject.save()
        setPatientSession(request, cardNo)
        username = getsession(request)
        userObj = pathologist_account.objects.filter(username=username)
        fullName = ""
        if len(userObj) > 0:
            fullName = userObj[0].fullName

        return render(request, "analyzeImage.html", {"pCardNo": cardNo, "fullName": fullName})
    except Exception as e:
        print(e)


def getPatientInfor(request):
    cardNo = request.GET.get("cardNo")
    patient_info_object = patient_info.objects.filter(patient_id=cardNo)
    patient_info_object = patient_info_object[0]
    result = "Patient ID :"+cardNo+"<br>"
    result += "Full name : " + \
        str(patient_info_object.patient_fullName)+"<br>"
    result += "Date of birth : " + \
        str(patient_info_object.patient_dateOfBirth)+"<br>"
    result += "Address : "+str(patient_info_object.patient_address)+"<br>"
    result += "Phone NO : "+str(patient_info_object.patient_phone)+"<br>"
    return HttpResponse(result)


def saveRecord(request):
    try:
        cardNo = request.POST.get("cardNo")
        fullName = request.POST.get("fullName")
        address = request.POST.get("address")
        phone = request.POST.get("phone")
        dateOfBirth = request.POST.get("dateOfBirth")
        sampleDate = request.POST.get("todayDate")
        sampleTime = request.POST.get("todayTime")
        sampleType = request.POST.get("sampleType")
        cancerName = request.POST.get("cancerName")
        cancerType = request.POST.get("cancerType")

        cancerSubType = request.POST.get("cancerSubType")
        cancerGrade = request.POST.get("cancerGrade")
        pathologistRemark = request.POST.get("remark")
        img = request.POST.get("canvasImage")
        cancerType = "ctype"
        cancerSubType = "ctype"
        cancerGrade = "ctype"
        img = img.replace('data:image/png;base64,', '')
        img = img.replace(' ', '+')
        img = base64.b64decode(img)
        # getting recording pathologist name
        username = getsession(request)
        userObj = pathologist_account.objects.filter(username=username)
        if len(userObj) > 0:
            registering_pathologist = userObj[0].fullName

            # Checking whether the patient is registered previously
            patientObj = patient_info.objects.filter(patient_id=cardNo)
            if len(patientObj) > 0:
                patientObj = patientObj[0]
                # register patient test Details
                if img != None:
                    result = registering_test_detials(cardNo, registering_pathologist, sampleType, cancerName,
                                                      cancerType, cancerSubType, cancerGrade, pathologistRemark, sampleDate, sampleTime, img)
                    return HttpResponse(result)
                else:
                    return HttpResponse("4")
            else:
                # registering new patient
                patientObj = patient_info(patient_id=cardNo, patient_fullName=fullName,
                                          patient_address=address, patient_phone=phone, patient_dateOfBirth=dateOfBirth)
                patientObj.save()
                result = registering_test_detials(cardNo, registering_pathologist, sampleType, cancerName,
                                                  cancerType, cancerSubType, cancerGrade, pathologistRemark, sampleDate, sampleTime, img)
                return HttpResponse(result)
        else:
            return HttpResponse("5")
    except Exception as e:
        print(e)
        return HttpResponse("6")


def registering_test_detials(cardNo, recording_pathologist_name, sampleType, cancerName, cancerType, cancerSubType, cancerGrade, pathologistRemark, sampleDate, sampleTime, file):
    try:
        patientObj = patient_info.objects.filter(patient_id=cardNo)
        if len(patientObj) > 0:
            patientObj = patientObj[0]
        testDetailsObj = patient_testDetails(patient=patientObj, recording_pathologist_name=recording_pathologist_name, sample_type=sampleType, cancer_name=cancerName, cancer_type=cancerType,
                                             cancer_subType=cancerSubType, cancer_grade=cancerGrade, pathologist_remark=pathologistRemark, record_date=sampleDate, record_time=sampleTime, sample_imageFile="")
        testDetailsObj.save()
        obj = patient_testDetails.objects.latest('id')
        graphicsname = cancerName+"_"+cancerType+"_" + \
            cancerSubType+"_"+cancerGrade+"_"+str(hex(obj.id))+".png"
        path = ""
        if cancerName == "Breast":
            path = 'ACDMSApp/static/breast_cancer_images/'
        else:
            path = 'ACDMSApp/static/cervical_cancer_images/'
        status = handle_uploaded_file(file, path+graphicsname)
        if status:
            patient_testDetails.objects.filter(pk=obj.id).update(
                sample_imageFile=path+graphicsname)
            return "1"
        else:
            return "2"
    except Exception as e:
        print(e)
        return "3"


def handle_uploaded_file(file, filename):
    status = False
    try:
        f = open(filename, 'wb')
        f.write(file)
        f.close()
        return True
    except Exception as e:
        print(e)
        return False


def automaticPatientInfoFill(request):
    patient_info_value = {}
    try:
        cardNo = request.GET.get("cardNo")
        print(cardNo)
        patientObj = patient_info.objects.filter(patient_id=cardNo)
        if len(patientObj) > 0:
            patientObj = patientObj[0]
            patient_info_value["fullName"] = patientObj.patient_fullName
            patient_info_value["address"] = patientObj.patient_address
            patient_info_value["phone"] = patientObj.patient_phone
            patient_info_value["dateOfBirth"] = patientObj.patient_dateOfBirth
        return JsonResponse(patient_info_value)
    except Exception as e:
        print(e)
        return JsonResponse(patient_info_value)


def processingPapSmearImage(request):
    try:
        if request.POST.get("imageData"):
            img = request.POST.get("imageData")
            img = img.replace('data:image/png;base64,', '')
            img = img.replace(' ', '+')
            img = base64.b64decode(img)
            # saving file as image
            file = open("ACDMSApp/static/temp_image.bmp", 'wb')
            file.write(img)
            file.close()

            model = pap_smear_classifier_2()
            model.load_weights("ACDMSApp/static/papsmear_final_round_2.h5")
            img = cv2.imread("ACDMSApp/static/temp_image.bmp")[..., ::-1]
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            # img_array = image.img_to_array(img)
            img = np.array(img)/255
            img = img.reshape(1, 224, 224, 3)
            # img_batch = np.expand_dims(resized_arr, axis=0)

            # img_preprocessed = preprocess_input(img_batch)
            prediction = model.predict(img)
            y_classes = prediction.argmax(axis=-1)[0]
            labelMap = ["abnormal", "normal"]
            # labelMap = {'Type1': 0, 'Type2': 1,'Type3':2}
            # keys = [k for k, v in labelMap.items() if v == y_classes]
            result = labelMap[y_classes]

            return HttpResponse(result)
        else:
            return HttpResponse("1")
    except Exception as e:
        print(e)
        return HttpResponse("1")

############################################################


@csrf_exempt
def processImageBreastCancerBinary(request):
    try:
        if request.POST.get("imageData"):
            cancerType = request.POST.get("cancerName")
            if cancerType == 'b':
                return HttpResponse(breastCancerClassication(request))
            else:
                cervixClassificationType = request.POST.get("cervixAnalysingType")
                if cervixClassificationType == "cervixType":
                    return HttpResponse(cevixTypeClassifier(request))
                elif cervixClassificationType == "cervicalCancer":
                    return HttpResponse(cervicalCancerClassifier(request))
                elif cervixClassificationType == "papSmear":
                    return HttpResponse(papSmearClassisifier(request))
                else:
                    return HttpResponse("1")
        else:
            return HttpResponse("1")
    except Exception as e:
        print(e)
        return HttpResponse("1")

def breastCancerClassication(request):
    img = request.POST.get("imageData")
    img = img.replace('data:image/png;base64,', '')
    img = img.replace(' ', '+')
    img = base64.b64decode(img)
    # saving file as image
    file = open("ACDMSApp/static/temp_image.png", 'wb')
    file.write(img)
    file.close()

    model = breast_cancer_binary_classifier_model1()
    model.load_weights(
        "ACDMSApp/static/BreastCancerClassClassifier_round1.h5")
    img = cv2.imread("ACDMSApp/static/temp_image.png")[..., ::-1]
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("ACDMSApp/static/temp_imageBGR.png", img1)
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("ACDMSApp/static/temp_imageGRAY.png", img2)
    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("ACDMSApp/static/temp_imageRGB.png", img3)
    img = cv2.resize(
        img, (224, 224), interpolation=cv2.INTER_LINEAR)
    # img_array = image.img_to_array(img)
    img = np.array(img)/255
    img = img.reshape(1, 224, 224, 3)
    # img_batch = np.expand_dims(resized_arr, axis=0)

    # img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img)
    y_classes = prediction.argmax(axis=-1)[0]
    labelMap = ['benign', 'malignant']
    # keys = [k for k, v in labelMap.items() if v == y_classes]
    cancer_class = str(labelMap[y_classes])
    subClass = ""
    if cancer_class == "benign":
        subClass = getBenignSubClass(img)
    else:
        subClass = getMalignantSubClass(img)
    classHtmlString = "<ul>"
    prediction = prediction[0]
    for i in range(0, len(prediction)):
        if i == y_classes:
            classHtmlString += "<li><b>" + \
                str(int(prediction[i]*100))+"% " + \
                str(labelMap[i])+"</b></li>"
        else:
            classHtmlString += "<li>" + \
                str(int(prediction[i]*100))+"% " + \
                str(labelMap[i])+"</li>"
    classHtmlString += "</ul>"
    return ("Breast cancer"+","+classHtmlString+","+subClass)

def benignClassifierModel():
    model = Sequential()
    model.add(Conv2D(64, 3, padding="same",
              activation="relu", input_shape=(224, 224, 3)))
    model.add(MaxPool2D(2, 2))

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(64, 3, padding="same", activation="relu"))

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu',
              bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(Dense(64, activation="relu",
              bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(Dense(4, activation="softmax"))
    opt = Adam(lr=1e-5)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def getBenignSubClass(img):
    model = benignClassifierModel()
    model.load_weights("ACDMSApp/static/benignClassifier1.h5")
    prediction = model.predict(img)
    labels = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma']
    max_arg = prediction.argmax(axis=-1)[0]
    result = labels[max_arg]
    subClassHtmlString = "<ul>"
    prediction = prediction[0]
    for i in range(0, len(prediction)):
        if i == max_arg:
            subClassHtmlString += "<li><b>" + \
                str(int(prediction[i]*100))+"% " + \
                str(labels[i])+"</b></li>"
        else:
            subClassHtmlString += "<li>" + \
                str(int(prediction[i]*100))+"% " + \
                str(labels[i])+"</li>"
    subClassHtmlString += "</ul>"
    return subClassHtmlString


def malignantClassifierModel():
    model = Sequential()
    model.add(Conv2D(64, 3, padding="same",
              activation="relu", input_shape=(224, 224, 3)))
    model.add(MaxPool2D(2, 2))

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Conv2D(64, 3, padding="same", activation="relu"))

    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu',
              bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(Dense(64, activation="relu",
              bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(Dense(4, activation="softmax"))
    opt = Adam(lr=1e-5)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def getMalignantSubClass(img):
    model = malignantClassifierModel()
    model.load_weights("ACDMSApp/static/malignantClassifier1.h5")
    prediction = model.predict(img)
    labels = ['ductal', 'lobular', 'mucinous', 'papillary']
    max_arg = prediction.argmax(axis=-1)[0]
    result = labels[max_arg]
    subClassHtmlString = "<ul>"
    prediction = prediction[0]
    for i in range(0, len(prediction)):
        if i == max_arg:
            subClassHtmlString += "<li><b>" + \
                str(int(prediction[i]*100))+"% " + \
                str(labels[i])+"</b></li>"
        else:
            subClassHtmlString += "<li>" + \
                str(int(prediction[i]*100))+"% " + \
                str(labels[i])+"</li>"
    subClassHtmlString += "</ul>"
    return subClassHtmlString

def cervixTypeClassifierModel():
    model = Sequential()
    model.add(Conv2D(128, (4,4), activation='relu', input_shape=(150,150, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (8, 8), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(8, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(150, activation='relu'))
    model.add(Dense(3, activation = 'softmax'))
    return model
def cevixTypeClassifier(request):
    img = request.POST.get("imageData")
    img = img.replace('data:image/png;base64,', '')
    img = img.replace(' ', '+')
    img = base64.b64decode(img)
    # saving file as image
    file = open("ACDMSApp/static/temp_image.png", 'wb')
    file.write(img)
    file.close()

    img = cv2.imread("ACDMSApp/static/temp_image.png")
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("ACDMSApp/static/temp_imageBGR.png", img1)
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("ACDMSApp/static/temp_imageGRAY.png", img2)
    img3 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    cv2.imwrite("ACDMSApp/static/temp_imageRGB.png", img3)
    img = cv2.resize(
        img, (150, 150), interpolation=cv2.INTER_LINEAR)
    # img_array = image.img_to_array(img)
    img = np.array(img)/255
    img = img.reshape(1, 150, 150, 3)
    # img_batch = np.expand_dims(resized_arr, axis=0)

    # img_preprocessed = preprocess_input(img_batch)
    model = cervixTypeClassifierModel()
    model.load_weights("ACDMSApp/static/cervixTypeClassifierAcc87Val62.h5")
    prediction = model.predict(img)
    y_classes = prediction.argmax(axis=-1)[0]
    labelMap = ['Type_1', 'Type_2', 'Type_3']
    # keys = [k for k, v in labelMap.items() if v == y_classes]
    cancer_class = str(labelMap[y_classes])
    classHtmlString = "<ul>"
    prediction = prediction[0]
    for i in range(0, len(prediction)):
        if i == y_classes:
            classHtmlString += "<li><b>" + \
                str(int(prediction[i]*100))+"% " + \
                str(labelMap[i])+"</b></li>"
        else:
            classHtmlString += "<li>" + \
                str(int(prediction[i]*100))+"% " + \
                str(labelMap[i])+"</li>"
    classHtmlString += "</ul>"
    return ("Cervical cancer<br><i>Cervix Type</i>"+","+classHtmlString+",")
def cervicalCancerClassifierModel():
    model = Sequential()
    model.add(Conv2D(128, (4,4), activation='relu', input_shape=(150,150, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (8, 8), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(8, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(150, activation='relu'))
    model.add(Dense(3, activation = 'softmax'))
    return model
def cervicalCancerClassifier(request):
    img = request.POST.get("imageData")
    img = img.replace('data:image/png;base64,', '')
    img = img.replace(' ', '+')
    img = base64.b64decode(img)
    # saving file as image
    file = open("ACDMSApp/static/temp_image.png", 'wb')
    file.write(img)
    file.close()

    model = cervicalCancerClassifierModel()
    model.load_weights("ACDMSApp/static/cervicalCancerClassifieracc90Val80.h5")
    img = cv2.imread("ACDMSApp/static/temp_image.png")[..., ::-1]
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("ACDMSApp/static/temp_imageBGR.png", img1)
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("ACDMSApp/static/temp_imageGRAY.png", img2)
    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("ACDMSApp/static/temp_imageRGB.png", img3)
    img = cv2.resize(
        img, (150, 150), interpolation=cv2.INTER_LINEAR)
    # img_array = image.img_to_array(img)
    img = np.array(img)/255
    img = img.reshape(1, 150, 150, 3)
    # img_batch = np.expand_dims(resized_arr, axis=0)

    # img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img)
    y_classes = prediction.argmax(axis=-1)[0]
    labelMap = ['Adinocarcinoma', 'Squamous_cell_carcinoma', 'Precancer']
    # keys = [k for k, v in labelMap.items() if v == y_classes]
    cancer_class = str(labelMap[y_classes])
    classHtmlString = "<ul>"
    prediction = prediction[0]
    for i in range(0, len(prediction)):
        if i == y_classes:
            classHtmlString += "<li><b>" + \
                str(int(prediction[i]*100))+"% " + \
                str(labelMap[i])+"</b></li>"
        else:
            classHtmlString += "<li>" + \
                str(int(prediction[i]*100))+"% " + \
                str(labelMap[i])+"</li>"
    classHtmlString += "</ul>"
    return ("Cervical cancer<br><i>Cervical Cancer</i>"+","+classHtmlString+",")
def papSmearClassisifierModel():
    model = Sequential()
    model.add(Conv2D(128, (4,4), activation='relu', input_shape=(150,150, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (8, 8), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(8, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(150, activation='relu'))
    model.add(Dense(2, activation = 'softmax'))
    return model
def papSmearClassisifier(request):
    img = request.POST.get("imageData")
    img = img.replace('data:image/png;base64,', '')
    img = img.replace(' ', '+')
    img = base64.b64decode(img)
    # saving file as image
    file = open("ACDMSApp/static/temp_image.png", 'wb')
    file.write(img)
    file.close()

    model = papSmearClassisifierModel()
    model.load_weights("ACDMSApp/static/papSmearClassifierAcc93Val87.h5")
    img = cv2.imread("ACDMSApp/static/temp_image.png")[..., ::-1]
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("ACDMSApp/static/temp_imageBGR.png", img1)
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("ACDMSApp/static/temp_imageGRAY.png", img2)
    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("ACDMSApp/static/temp_imageRGB.png", img3)
    
    img = cv2.resize(
        img, (150, 150), interpolation=cv2.INTER_LINEAR)
    # img_array = image.img_to_array(img)
    img = np.array(img)/255
    img = img.reshape(1, 150, 150, 3)
    # img_batch = np.expand_dims(resized_arr, axis=0)

    # img_preprocessed = preprocess_input(img_batch)
    prediction = model.predict(img)
    y_classes = prediction.argmax(axis=-1)[0]
    labelMap = ['normal', 'abnormal']
    # keys = [k for k, v in labelMap.items() if v == y_classes]
    cancer_class = str(labelMap[y_classes])
    classHtmlString = "<ul>"
    prediction = prediction[0]
    for i in range(0, len(prediction)):
        if i == y_classes:
            classHtmlString += "<li><b>" + \
                str(int(prediction[i]*100))+"% " + \
                str(labelMap[i])+"</b></li>"
        else:
            classHtmlString += "<li>" + \
                str(int(prediction[i]*100))+"% " + \
                str(labelMap[i])+"</li>"
    classHtmlString += "</ul>"
    return ("Cervical cancer<br><i>Pap smear classification</i>"+","+classHtmlString+",")
