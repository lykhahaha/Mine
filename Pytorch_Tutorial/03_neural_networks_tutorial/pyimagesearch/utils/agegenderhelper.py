import numpy as np
import glob
import cv2
import os
import re
from scipy.io import loadmat
import shutil
import dlib
from imutils.face_utils import FaceAligner
from imutils import face_utils
import imutils
from imutils import paths
import progressbar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

class AgeGenderHelper:
    def __init__(self, config, deploy):
        # store config object and build age bins used for constructing class labeld
        self.config = config
        self.deploy = deploy
        self.age_bins = self.build_age_bins()

    def build_age_bins(self):
        # initialize list of age bins based on Adience dataset
        if self.config.DATASET == 'IOG':
            age_bins = [(0, 2), (3, 7), (8, 12), (13, 19), (20, 36), (37, 65), (66, np.inf)]
        elif self.config.DATASET == 'ADIENCE':
            age_bins = [(0, 2), (4, 6), (8, 13), (15, 20), (25, 32), (38, 43), (48, 53), (60, np.inf)]

        return age_bins

    def to_label(self, age, gender):
        # check to see if we should determine age label
        if self.config.DATASET_TYPE == 'age':
            return self.to_age_label(age)
        return self.to_gender_label(gender)

    def to_age_label(self, age):
        # initialize label
        label = None
        
        if self.config.DATASET == 'IOG':
            # decode age label
            age_label_dict = {'1': '0_2', '5': '3_7', '10': '8_12', '16': '13_19', '28': '20_36', '51': '37_65', '75': '66_inf'}
            label = age_label_dict[age]
        elif self.config.DATASET == 'ADIENCE':
            # get ages from (23, 43)
            age = re.findall(r'(\d+)', age)
            # age = age.replace('(', '').replace(')', '').split(', ')
            (age_lower, age_upper) = np.array(age, dtype='int')

            # loop over age bins
            for lower, upper in self.age_bins:
                # determine if age falls into current bin
                if age_lower >= lower and age_upper <= upper:
                    label = f'{lower}_{upper}'
                    break

        return label

    def to_gender_label(self, gender):
        if self.config.DATASET == 'IOG':
            return 0 if gender == '2' else 1
        elif self.config.DATASET == 'ADIENCE':
            return 0 if gender == 'm' else 1
    
    def build_oneoff_mappings(self, le):
        # sort class labels in ascending order and initialize one-off mappings
        classes = sorted(le.classes_, key=lambda x: int(x.split('_')[0]))
        one_off = {}

        # loop over index and name of sorted class labels
        for i, name in enumerate(classes):
            # determine index of current class label in label encoder unordered list,
            # then initialize index of previous and next age groups adjacent to current label
            current = np.where(le.classes_ == name)[0][0]
            prev, next = -1, -1

            # check to see if we should compute previous adjacent age group
            if i > 0:
                prev = np.where(le.classes_ == classes[i-1])[0][0]

            # check to see if we compute next adjacent age group
            if i < len(classes) - 1:
                next = np.where(le.classes_ == classes[i+1])[0][0]

                # construct tuple that consists of current age bracket, the prev age bracket, next age bracket
            one_off[current] = (current, prev, next)

        return one_off

    def build_paths_and_labels_adience(self):
        # initialize list of image paths and labels
        cross_paths, cross_labels, cross_paths_frontal, cross_labels_frontal = [], [], [], []

        # grab paths to folds file
        fold_paths = os.path.sep.join([self.config.LABELS_PATH, 'fold_[0-9]_data.txt'])
        fold_paths_frontal = os.path.sep.join([self.config.LABELS_PATH, 'fold_frontal_*_data.txt'])
        fold_paths = glob.glob(fold_paths)
        fold_paths_frontal = glob.glob(fold_paths_frontal)
        fold_paths.sort()
        fold_paths_frontal.sort()

        # loop over folds paths
        for fold_path, fold_path_frontal in zip(fold_paths, fold_paths_frontal):
            paths, labels, paths_frontal, labels_frontal = [], [], [], []
            # load contents of folds file, skipping header
            rows = open(fold_path).read().strip().split('\n')[1:]
            rows_frontal = open(fold_path_frontal).read().strip().split('\n')[1:]

            # loop over rows
            for row in rows:
                # unpack needed components of row
                user_id, image_path, face_id, age, gender = row.split('\t')[:5]

                if self.config.DATASET_TYPE == 'age' and age[0] != '(':
                    continue

                if self.config.DATASET_TYPE == 'gender' and gender not in ('f', 'm'):
                    continue
                
                # construct path to input image and build class label
                p = os.path.sep.join([self.config.IMAGES_PATH, user_id, f'landmark_aligned_face.{face_id}.{image_path}'])
                label = self.to_label(age, gender)

                if label is None:
                    continue
                
                paths.append(p)
                labels.append(label)

            cross_paths.append(paths)
            cross_labels.append(labels)

            # loop over rows frontal
            for row_frontal in rows_frontal:
                # unpack needed components of row
                user_id, image_path, face_id, age, gender = row_frontal.split('\t')[:5]

                if self.config.DATASET_TYPE == 'age' and age[0] != '(':
                    continue

                if self.config.DATASET_TYPE == 'gender' and gender not in ('f', 'm'):
                    continue
                
                # construct path to input image and build class label
                p = os.path.sep.join([self.config.IMAGES_PATH, user_id, f'landmark_aligned_face.{face_id}.{image_path}'])
                label = self.to_label(age, gender)

                if label is None:
                    continue
                
                paths_frontal.append(p)
                labels_frontal.append(label)

            cross_paths_frontal.append(paths_frontal)
            cross_labels_frontal.append(labels_frontal)
        
        for i, (test_paths, test_labels, test_paths_frontal, test_labels_frontal) in enumerate(zip(cross_paths, cross_labels, cross_paths_frontal, cross_labels_frontal)):
            train_paths, train_labels = [], []
            for j in range(len(fold_paths)):
                if j != i:
                    train_paths.extend(cross_paths[j])
                    train_labels.extend(cross_labels[j])

            yield train_paths, train_labels, test_paths, test_labels, test_paths_frontal, test_labels_frontal
    
    def build_paths_and_labels_iog(self):
        # initialize list of image paths and labels
        paths, face_coords, labels = [], [], []

        # grab all person data files 
        person_data_paths = os.path.sep.join([self.config.BASE_PATH, '*', 'PersonData.txt'])
        person_data_paths = glob.glob(person_data_paths)

        # loop over person data files
        for person_data_path in person_data_paths:
            # get folder containing person data file
            folder = os.path.sep.join([p for p in person_data_path.split(os.path.sep)[:-1]])

            # load contents of file
            rows = open(person_data_path).read().strip().split('\n')
            
            # loop over rows
            for i, row in enumerate(rows):
                # check to see the line is image name, if so append it to image paths
                if 'jpg' in row:
                    if i > 0:
                        face_coords.append(face_coord)
                        labels.append(label)
                    paths.append(os.path.sep.join([folder, row]))

                    # initialize lists containing face coords and age, gender label of each image
                    face_coord, label = [], []
                
                else:
                    # get face coords and append to face corrds list
                    face_coord.append([int(c) for c in row.split()[:4]])
                    
                    # get age and gender and append to list
                    age, gender = row.split()[4:]
                    label.append(self.to_label(age, gender))

                if i == len(rows) - 1:
                    face_coords.append(face_coord)
                    labels.append(label)

        return paths, face_coords, labels

    def build_paths_and_labels_iog_test(self):
        # initialize list of image paths and labels
        paths, face_centers, labels = [], [], []

        mat = loadmat(self.config.MAT_TEST_PATH)

        mat_paths, mat_centers, mat_labels = [p[0].split('\\')[-1] for p in mat['tecoll'][0][0]['name'][0]], mat['tecoll'][0][0]['facePosSize'][:, [4, 5]], mat['tecoll'][0][0]['ageClass'][:, 0] if self.config.DATASET_TYPE == 'age' else mat['tecoll'][0][0]['genClass'][:, 0]

        paths_info = {}

        for i, mat_path in enumerate(mat_paths):
            path_info = paths_info.get(mat_path, {'face_centers': [], 'labels': []})
            path_info['face_centers'].append(mat_centers[i].tolist())
            path_info['labels'].append(self.to_label(str(mat_labels[i]), str(mat_labels[i])))
            paths_info[mat_path] = path_info

        for unique_path, path_info in paths_info.items():
            path = os.path.sep.join([self.config.BASE_PATH, '*', f'{unique_path}'])
            unique_path = glob.glob(path)[0]
            paths.append(unique_path)

            face_centers.append(path_info['face_centers'])
            labels.append(path_info['labels'])
        
        return paths, face_centers, labels

    def build_face_images_iog(self, folder_path, is_test):
        class Counter(dict):
            def __missing__(self, key):
                return 0

        # remove all age and gender path if exist
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)

        # get image paths, face coordiantes and labels
        if is_test == True:
            print('[INFO] building multi-face image paths and labels for testing...')
            image_paths, face_coords, labels = self.build_paths_and_labels_iog_test()
        else:
            print('[INFO] building multi-face image paths and labels for training...')
            image_paths, face_coords, labels = self.build_paths_and_labels_iog()
        
        # initialize dlib's face detector (HOG-based), then create facial landmark predictor and face aligner
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.deploy.DLIB_LANDMARK_PATH)
        fa = FaceAligner(predictor)

        # construct progress bar
        if is_test == True:
            widgets = [f'Serializing faces for testing: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        else:
            widgets = [f'Serializing faces for training: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(labels), widgets=widgets).start()

        # initialize counter for file name
        file_counter = Counter()

        for i, (image_path, face_coord, label) in enumerate(zip(image_paths, face_coords, labels)):
            # load image and get its dimension
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            scale = w/1024

            # resize, convert image to gray scale and pass it through detector
            image = imutils.resize(image, width=1024)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            # get index after sorting face coordinations
            face_coord_idx = sorted(range(len(face_coord)), key=lambda k: (face_coord[k], face_coord[k][0], face_coord[k][1]))

            # get the center point of left and right eye points
            center_face_coord, label = np.array(face_coord)[face_coord_idx], np.array(label)[face_coord_idx]

            if is_test is False:
                even_col_center = np.apply_along_axis(lambda col:(col[0] + col[2])/2, 1, center_face_coord)
                odd_col_center = np.apply_along_axis(lambda col:(col[1] + col[3])/2, 1, center_face_coord)
                center_face_coord = np.vstack((even_col_center, odd_col_center)).T
            
            # loop over the rects
            for rect in rects:
                # change the rect detector detects to bounding box
                x, y, w, h = [x*scale for x in face_utils.rect_to_bb(rect)]

                # loop over all center points to check which center point is in bounding box, if so apply Facial Alignment, then save it based on its label
                for j, center in enumerate(center_face_coord):
                    if x < center[0] < x+w and y < center[1] < y+h:
                        # determine facial landmarks for face region, then align face
                        shape = predictor(gray, rect)
                        face = fa.align(image, gray, rect)

                        # initialize path to save the face detected
                        path_face = os.path.sep.join([folder_path, f'{label[j]}'])

                        if not os.path.exists(path_face):
                            os.makedirs(path_face)

                        cv2.imwrite(os.path.sep.join([path_face, f'{file_counter[label[j]]:05d}.jpg']), face)

                        file_counter[label[j]] += 1
            
            pbar.update(i)

        pbar.finish()

    def build_paths_and_labels_iog_preprocessed(self):
        self.build_face_images_iog(self.config.DATASET_PATH, is_test=False)
        train_paths = list(paths.list_images(self.config.DATASET_PATH))
        train_labels = [p.split(os.path.sep)[-2] for p in train_paths]
        
        self.build_face_images_iog(self.config.DATASET_TEST_PATH, is_test=True)
        test_paths = list(paths.list_images(self.config.DATASET_TEST_PATH))
        test_labels = [p.split(os.path.sep)[-2] for p in test_paths]

        return train_paths, train_labels, test_paths, test_labels


    @staticmethod
    def build_mapping_to_iog_labels():
        return np.array([0, 1, 2, 3, 3, 4, 5, 6])

    @staticmethod
    def visualize_age(age_preds, le):
		# initialize the canvas and sort the predictions according
		# to their probability
        canvas = np.zeros((250, 310, 3), dtype="uint8")
        idxs = np.argsort(age_preds)[::-1]

		# loop over the age predictions in ascending order
        for (i, j) in enumerate(idxs):
			# construct the text for the prediction
			#age_label = le.inverse_transform(j) # Python 2.7
            age_label = le.classes_[j]# .decode("utf-8")
            age_label = age_label.replace("_", "-")
            age_label = age_label.replace("-inf", "+")

			# draw the label + probability bar on the canvas
            w = int(age_preds[j] * 300) + 5
            cv2.rectangle(canvas, (5, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, f'{age_label}: {age_preds[j] * 100:.2f}%', (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
		
		# return the visualization
        return canvas

    @staticmethod
    def visualize_gender(gender_preds, le):
		# initialize the canvas and sort the predictions according
		# to their probability
        canvas = np.zeros((100, 310, 3), dtype="uint8")
        idxs = np.argsort(gender_preds)[::-1]

        # loop over the gender predictions in ascending order
        for (i, j) in enumerate(idxs):
			# construct the text for the prediction
            gender = le.classes_[j]
            gender = "Male" if gender == 0 else "Female"

            # draw the label + probability bar on the canvas
            w = int(gender_preds[j] * 300) + 5
            cv2.rectangle(canvas, (5, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, f'{gender}: {gender_preds[j] * 100:.2f}%', (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

		# return the canvas
        return canvas

    @staticmethod
    def visualize_video(age_pred, gender_pred, age_le, gender_le, canvas, top_left_pts):
        age_idx, gender_idx = np.argmax(age_pred), np.argmax(gender_pred)
        age_prob, gender_prob = age_pred[age_idx], gender_pred[gender_idx]

        age_label = age_le.classes_[age_idx]
        age_label = age_label.replace("_", "-")
        age_label = age_label.replace("-inf", "+")

        gender_label = "Male" if gender_le.classes_[gender_idx] == 0 else "Female"

        cv2.putText(canvas, f'{age_label}: {age_prob:.2f}/ {gender_label}: {gender_prob:.2f}', (top_left_pts[0], top_left_pts[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        return canvas

    @staticmethod
    def plot_confusion_matrix_from_data(config, y_true, y_pred, le, save_path, figsize=(12, 9)):
        if config.DATASET_TYPE == 'age':
            y_true, y_pred, labels = le.inverse_transform(y_true), le.inverse_transform(y_pred), le.classes_
            figsize=(12, 9)
        elif config.DATASET_TYPE == 'gender':
            labels = np.array(['Male', 'Female'])
            y_true = labels[y_true]
            y_pred = labels[y_pred]
            figsize=(5, 5)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        cm = pd.DataFrame(cm, index=labels, columns=labels)
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=annot, annot_kws={"size": 12}, linewidths=0.5, fmt='', ax=ax, linecolor='w')
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)