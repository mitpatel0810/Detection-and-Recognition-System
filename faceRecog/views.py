from django.shortcuts import render, redirect, render_to_response
import cv2
import datetime
import time
import subprocess
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from . import dataset_fetch as df
from . import cascade as casc
from PIL import Image
import os
import urllib2
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
from records.models import Records
from django.http import HttpResponse

# from settings import BASE_DIR
# Create your views here.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def index(request):
    return render(request, 'index.html')


def unknown(request):
    path = BASE_DIR + "/static/unknown/gray"
    img_list = os.listdir(path)
    return render_to_response('unknown.html', {'images': img_list, 'BASE_DIR': BASE_DIR})


def errorImg(request):
    return render(request, 'error.html')


def dataset(request):
    data = Records.objects.all()

    stu = {
        "id": data
    }

    return render_to_response("dataset.html", stu)

    # return render(request, 'dataset.html')


def live(request):
    return render(request, 'live.html')


def livecam(request):


    faceDetect = cv2.CascadeClassifier(BASE_DIR + '/ml/haarcascade_frontalface_default.xml')
    cam = cv2.VideoCapture("https://192.168.1.6:8080/video")
    while (True):
        ret, img = cam.read()
        if (cv2.waitKey(1) == ord('q')):
            ts_time = time.time()
            s_time = datetime.datetime.fromtimestamp(ts_time).strftime('%d%m%Y%H%M%S')
            cv2.imwrite(BASE_DIR + '/static/live_color_img/' + str(s_time) + '.jpg', img)
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Live Cam", img)

        #cv2.imwrite(BASE_DIR + '/static/live_color_img/' + 'last seen live' + '.jpg',img)
        #with open('static/live_color_img/' + 'last seen live' + '.jpg', 'r') as image_file:
        #    response = HttpResponse(image_file.read(), content_type='image/jpg')
        #    response['Content-Disposition'] = 'inline; filename=%s' % 'image.jpg'
        #    return response
        #image_file.close

    cv2.destroyAllWindows()

    return redirect('/')


def create_dataset(request):
    # print request.POST
    userId = request.POST['userId']
    print (cv2.__version__)
    # Detect face
    # Creating a cascade image classifier
    # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    faceDetect = cv2.CascadeClassifier(BASE_DIR + '/ml/haarcascade_frontalface_default.xml')
    # camture images from the webcam and process and detect the face
    # takes video capture id, for webcam most of the time its 0.
    cam = cv2.VideoCapture("https://192.168.1.6:8080/video")

    # Our identifier
    # We will put the id here and we will store the id with a face, so that later we can identify whose face it is
    id = userId
    # Our dataset naming counter
    sampleNum = 0
    # Capturing the faces one by one and detect the faces and showing it on the window
    while (True):
        # Capturing the image
        # cam.read will return the status variable and the captured colored image
        ret, img = cam.read()
        # the returned img is a colored image but for the classifier to work we need a greyscale image
        # to convert
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # To store the faces
        # This will detect all the images in the current frame, and it will return the coordinates of the faces
        # Takes in image and some other parameter for accurate result
        faces = faceDetect.detectMultiScale(gray, 1.2, 5)
        # In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
        for (x, y, w, h) in faces:
            # Whenever the program captures the face, we will write that is a folder
            # Before capturing the face, we need to tell the script whose face it is
            # For that we will need an identifier, here we call it id
            # So now we captured a face, we need to write it in a file
            sampleNum = sampleNum + 1
            # Saving the image dataset, but only the face part, cropping the rest
            cv2.imwrite(BASE_DIR + '/ml/dataset/user.' + str(id) + '.' + str(sampleNum) + '.jpg',
                        gray[y:y + h, x:x + w])
            # @params the initial point of the rectangle will be x,y and
            # @params end point will be x+width and y+height
            # @params along with color of the rectangle
            # @params thickness of the rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 100 millisecond
            cv2.waitKey(250)

        # Showing the image in another window
        # Creates a window with window name "Face" and with the image img
        cv2.imshow("Face", img)
        # Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        cv2.waitKey(1)
        # To get out of the loop
        if (sampleNum > 35):
            break
    # releasing the cam
    cam.release()
    # destroying all the windows
    cv2.destroyAllWindows()

    return redirect('/')


def trainer(request):
    '''
        In trainer.py we have to get all the samples from the dataset folder,
        for the trainer to recognize which id number is for which face.

        for that we need to extract all the relative path
        i.e. dataset/user.1.1.jpg, dataset/user.1.2.jpg, dataset/user.1.3.jpg
        for this python has a library called os
    '''
    import os
    from PIL import Image
    import cv2
    import numpy as np

    # Creating a recognizer to train
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # recognizer=cv2.face.createLBPHFaceRecognizer()
    # Path of the samples
    path = BASE_DIR + '/ml/dataset'

    # To get all the images, we need corresponing id
    def getImagesWithID(path):
        # create a list for the path for all the images that is available in the folder
        # from the path(dataset folder) this is listing all the directories and it is fetching the directories from each and every pictures
        # And putting them in 'f' and join method is appending the f(file name) to the path with the '/'
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # concatinate the path with the image name
        # print imagePaths

        # Now, we loop all the images and store that userid and the face with different image list
        faces = []
        Ids = []
        for imagePath in imagePaths:
            # First we have to open the image then we have to convert it into numpy array
            faceImg = Image.open(imagePath).convert('L')  # convert it to grayscale
            # converting the PIL image to numpy array
            # @params takes image and convertion format
            faceNp = np.array(faceImg, 'uint8')
            # Now we need to get the user id, which we can get from the name of the picture
            # for this we have to slit the path() i.e dataset/user.1.7.jpg with path splitter and then get the second part only i.e. user.1.7.jpg
            # Then we split the second part with . splitter
            # Initially in string format so hance have to convert into int format
            ID = int(os.path.split(imagePath)[-1].split('.')[
                         1])  # -1 so that it will count from backwards and slipt the second index of the '.' Hence id
            # Images
            faces.append(faceNp)
            # Label
            Ids.append(ID)
            # print ID
            cv2.imshow("training", faceNp)
            cv2.waitKey(10)
        return np.array(Ids), np.array(faces)

    # Fetching ids and faces
    ids, faces = getImagesWithID(path)

    # Training the recognizer
    # For that we need face samples and corresponding labels
    recognizer.train(faces, ids)

    # Save the recogzier state so that we can access it later
    recognizer.save(BASE_DIR + '/ml/recognizer/trainingData.yml')
    cv2.destroyAllWindows()

    return redirect('/')


def detect(request):
    faceDetect = cv2.CascadeClassifier(BASE_DIR + '/ml/haarcascade_frontalface_default.xml')

    cam = cv2.VideoCapture("https://192.168.1.6:8080/video")
    # creating recognizer
    rec = cv2.face.LBPHFaceRecognizer_create();
    # loading the training data
    rec.read(BASE_DIR + '/ml/recognizer/trainingData.yml')
    getId = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    userId = 0
    # start_time = time.time()
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            getId, conf = rec.predict(gray[y:y + h, x:x + w])  # This will predict the id of the face

            if conf < 35:
                userId = getId

                #updating records
                ts_time = time.time()
                ts = datetime.datetime.fromtimestamp(ts_time).strftime('%d-%m-%Y %H:%M:%S')
                Records.objects.filter(id=userId).update(last_visit=ts)
                visit = Records.objects.values_list('visits', flat=True).get(id=userId)
                Records.objects.filter(id=userId).update(visits=visit + 1)

                #saving image to dir
                s_time = datetime.datetime.fromtimestamp(ts_time).strftime('%d%m%Y%H%M%S')
                temp_d_img = cv2.putText(img, "Detected", (x, y + h), font, 2, (0, 255, 0), 2)
                cv2.imwrite(BASE_DIR + '/static/known/' + str(s_time) + '.jpg', temp_d_img)

                #sending above image
                if internet_on():
                    print 'Device is Connected to Internet'
                    mail_this(userId, ts_time)
                else:
                    print 'Device is Disconnected to Internet'
            else:
                temp_img = cv2.putText(img, "Unknown", (x, y + h), font, 2, (0, 0, 255), 2)
                break

        cv2.imshow("Face", img)
        if (cv2.waitKey(1) == ord('q')):

            #to savethe gray image
            ts_time = time.time()
            ts = datetime.datetime.fromtimestamp(ts_time).strftime('%d%m%Y%H%M%S')
            for (x, y, w, h) in faces:
                cv2.imwrite(BASE_DIR + '/static/unknown/gray/' + str(ts) + '.jpg',
                            gray[y:y + h, x:x + w])

            #to save the colored image
            cv2.imwrite(BASE_DIR + '/static/unknown/' + str(ts) + '.jpg', temp_img)

            #to mail above image
            if internet_on():
                print 'Device is Connected to Internet'
                mail_this(-1,ts_time)
            else:
                print 'Device is Disconnected to Internet'
            break

        elif (userId != 0):
            cv2.waitKey(1000)
            cam.release()
            cv2.destroyAllWindows()
            return redirect('/records/details/' + str(userId))

    cam.release()
    cv2.destroyAllWindows()
    return redirect('/')


def mail_this(checkid, ts_time):

    strFrom = 'my.prime.bot@gmail.com'
    strTo = 'royal25796@gmail.com'

    msgRoot = MIMEMultipart('related')
    ts = datetime.datetime.fromtimestamp(ts_time).strftime('%d%m%Y%H%M%S')

    if checkid==-1:   #for unknown
        msgRoot['Subject'] = 'Unknown Person Encountered'
        file = 'static/unknown/' + str(ts) + '.jpg'
    else:            #for known
        msgRoot['Subject'] = 'Person Encountered'
        file = 'static/known/' + str(ts) + '.jpg'

    msgRoot['From'] = strFrom
    msgRoot['To'] = strTo
    msgRoot.preamble = 'This is a multi-part message in MIME format.'

    msgAlternative = MIMEMultipart('alternative')
    msgRoot.attach(msgAlternative)

    msgText = MIMEText('This is the alternative plain text message.')
    msgAlternative.attach(msgText)

    s_time = datetime.datetime.fromtimestamp(ts_time).strftime('%d-%m-%Y %H:%M:%S')

    if checkid==-1:  #for unknown
        msgText = MIMEText('<h2>We have encountered an unknown person in Premises </h2>'
                           '<h3>Name: ---</h3><h4>Occupation: ---<br>Residence: ---</h4><h5>Seen at: ' + s_time + '</h5><h5>Total Visits: ---</h5><img src="cid:image1">',
                           'html')
    else:           #for known
        msgText = MIMEText('<h2>We have encountered this person in Premises </h2>'
                           '<h3>Name: ' + Records.objects.values_list('first_name', flat=True).get(
            id=checkid) + ' ' + Records.objects.values_list('last_name', flat=True).get(id=checkid) +
                           '</h3><h4>Occupation: ' + Records.objects.values_list('occupation', flat=True).get(
            id=checkid) +
                           '<br>Residence: ' + Records.objects.values_list('residence', flat=True).get(
            id=checkid) + ', ' + Records.objects.values_list('country', flat=True).get(id=checkid) + '</h4>' +
                           '<h5>Seen at: ' + s_time + '</h5>'
                           + '<h5>Total Visits: ' + str(
            Records.objects.values_list('visits', flat=True).get(id=checkid)) + '</h5>'
                           + '<img src="cid:image1">'
                           , 'html')

    msgAlternative.attach(msgText)

    fp = open(file, 'rb')
    msg_d_Image = MIMEImage(fp.read())
    fp.close()

    msg_d_Image.add_header('Content-ID', '<image1>')
    msgRoot.attach(msg_d_Image)

    import smtplib

    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.starttls()
    smtp.login('my.prime.bot@gmail.com', 'Optimus Prime')
    smtp.sendmail(strFrom, strTo, msgRoot.as_string())
    smtp.quit()

    # print("Mail sent @: " + ts)


def internet_on():
    try:
        urllib2.urlopen('http://216.58.192.142', timeout=1)
        return True
    except urllib2.URLError as err:
        return False


def eigenTrain(request):
    path = BASE_DIR + '/ml/dataset'

    # Fetching training and testing dataset along with their image resolution(h,w)
    ids, faces, h, w = df.getImagesWithID(path)
    print ('features' + str(faces.shape[1]))
    # Spliting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(faces, ids, test_size=0.25, random_state=42)
    # print ">>>>>>>>>>>>>>> "+str(y_test.size)
    n_classes = y_test.size
    target_names = ['Manjil Tamang', 'Marina Tamang', 'Anmol Chalise']
    n_components = 15
    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    t0 = time.time()

    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)

    print("done in %0.3fs" % (time.time() - t0))
    eigenfaces = pca.components_.reshape((n_components, h, w))
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time.time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time.time() - t0))

    # #############################################################################
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time.time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time.time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    # #############################################################################
    # Quantitative evaluation of the model quality on the test set

    print ("Predicting people's names on the test set")
    t0 = time.time()
    y_pred = clf.predict(X_test_pca)
    print("Predicted labels: ", y_pred)
    print("done in %0.3fs" % (time.time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))

    # print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    # #############################################################################
    # Qualitative evaluation of the predictions using matplotlib

    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())

    # plot the gallery of the most significative eigenfaces
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)
    # plt.show()

    '''
        -- Saving classifier state with pickle
    '''
    svm_pkl_filename = BASE_DIR + '/ml/serializer/svm_classifier.pkl'
    # Open the file to save as pkl file
    svm_model_pkl = open(svm_pkl_filename, 'wb')
    pickle.dump(clf, svm_model_pkl)
    # Close the pickle instances
    svm_model_pkl.close()

    pca_pkl_filename = BASE_DIR + '/ml/serializer/pca_state.pkl'
    # Open the file to save as pkl file
    pca_pkl = open(pca_pkl_filename, 'wb')
    pickle.dump(pca, pca_pkl)
    # Close the pickle instances
    pca_pkl.close()

    plt.show()

    return redirect('/')


def detectImage(request):
    userImage = request.FILES['userImage']

    svm_pkl_filename = BASE_DIR + '/ml/serializer/svm_classifier.pkl'

    svm_model_pkl = open(svm_pkl_filename, 'rb')
    svm_model = pickle.load(svm_model_pkl)
    # print "Loaded SVM model :: ", svm_model

    pca_pkl_filename = BASE_DIR + '/ml/serializer/pca_state.pkl'

    pca_model_pkl = open(pca_pkl_filename, 'rb')
    pca = pickle.load(pca_model_pkl)
    # print pca

    '''
    First Save image as cv2.imread only accepts path
    '''
    im = Image.open(userImage)


    # im.show()
    imgPath = BASE_DIR + '/ml/uploadedImages/' + str(userImage)
    im.save(imgPath, 'JPEG')

    '''
    Input Image
    '''
    try:
        inputImg = casc.facecrop(imgPath)
        #inputImg.show()
    except:
        print("No face detected, or image not recognized")
        return redirect('/error_image')

    imgNp = np.array(inputImg, 'uint8')
    # Converting 2D array into 1D
    imgFlatten = imgNp.flatten()
    # print imgFlatten
    # print imgNp
    imgArrTwoD = []
    imgArrTwoD.append(imgFlatten)
    # Applyting pca
    img_pca = pca.transform(imgArrTwoD)
    # print img_pca

    pred = svm_model.predict(img_pca)
    print(svm_model.best_estimator_)
    print (pred[0])

    return redirect('/records/details/' + str(pred[0]))
