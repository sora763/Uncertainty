import os

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

import tools as tl

class Evaluate(object):
    def __init__(self,
                 data,
                 arch,
                 method,
                 patch_size,
                 stage):
        self.data = data
        self.patch_size = patch_size
        self.arch = arch
        self.method = method
        self.model_name = self.arch
        self.stage=stage

    def evaluate_one_image(self, y_true, y_d, labels):
        """
        evaluate segmentation result, using confusion_matrix
        y_true: 2d array,
        y_d: 2d array,
        labels: the labels dataset contains,
                ips -> [1, 2, 3], melanoma -> [0, 1]
        oor is ignored.
        """
        y_true = y_true.flatten()
        y_d = y_d.flatten()
        mat = confusion_matrix(y_true, y_d, labels=labels)
        jaccard = []
        dice = []
        tpr = []
        tnr = []
        acc = []
        class_j = np.zeros((3,))
        for i in range(len(labels)):
            if mat[i, :].sum() == 0:
                continue
            elif len(labels) == 2 and i == 0:
                continue
            tp = mat[i, i]
            tn = mat.sum() - (mat[i, :].sum() + mat[:, i].sum() - mat[i, i])
            fp = mat[:, i].sum() - mat[i, i]
            fn = mat[i, :].sum() - mat[i, i]
            jaccard.append(tp / float(tp + fp + fn))
            class_j[i] = (tp / float(tp + fp + fn))
            dice.append(2 * tp / float(2 * tp + fp + fn))
            tpr.append(tp / float(tp + fn))
            tnr.append(tn / float(fp + tn))
            acc.append((tp + tn) / float(tp + tn + fp + fn))

        jaccard = sum(jaccard) / len(jaccard)
        dice = sum(dice) / len(dice)
        tpr = sum(tpr) / len(tpr)
        tnr = sum(tnr) / len(tnr)
        acc = sum(acc) / len(acc)
        return jaccard, dice, tpr, tnr, acc, class_j


    def get_ytrue(self, mask_array, data):
        if self.data == "ips":
            height, width, _ = mask_array.shape
            good_label = ((mask_array[:,:,0] == 255)&
                          (mask_array[:,:,1] == 0)&
                          (mask_array[:,:,2] == 0)
                         ) * np.ones((height, width)) * 1
            bad_label = ((mask_array[:,:,0] == 0)&
                         (mask_array[:,:,1] == 255)&
                         (mask_array[:,:,2] == 0)
                        ) * np.ones((height, width)) * 2
            bgd_label = ((mask_array[:,:,0] == 0)&
                         (mask_array[:,:,1] == 0)&
                         (mask_array[:,:,2] == 255)
                        ) * np.ones((height, width)) * 3
            y_true = good_label + bad_label + bgd_label
            return y_true
        else:
            height, width = mask_array.shape
            y_true = (mask_array == 255) * np.ones((height, width))
            return y_true

    def get_label_list(self):
        if self.data == 'ips':
            return [1, 2, 3]
        else:
            return [0, 1]

    def evaluate(self):
        labels = self.get_label_list()
        all_j = []
        all_d = []
        all_tp = []
        all_tn = []
        all_acc = []
        for d_num in [1,2,3,4,5]:
            print(self.model_name, str(self.patch_size),"dataset_%d"%d_num)
            d_path = tl.get_save_path(self.data, self.method, self.model_name, self.patch_size, d_num)
            _, mask_path = tl.data_path_load(self.data, 'test', d_num)

            if self.stage == "stage2":
                d_path = d_path+"stage2/"

            elif self.stage == "stage3":
                d_path = d_path+"stage2/stage3/"

            elif self.stage == "stage4":
                d_path = d_path+"stage2/stage3/stage4/"
            with open(d_path + "image_evaluate.txt", "w") as file:
                title = ["<<", self.model_name, ">>"]
                title = " ".join(title)
                file.write(title + "\n")
                file.write("image_name jac. dice tpr tnr acc. \n")
            jaccard = []
            dice = []
            tpr = []
            tnr = []
            acc = []
            N = 1
            for m_path in mask_path:
                mask_name = os.path.basename(m_path)[:-4]
                y_d = np.array(Image.open(d_path + "label/%s.png"%(mask_name)), int)
                y_true = np.array(Image.open(m_path), int)
                if self.data == "ips":
                    mask_array = y_true[:,:,:3]
                else:
                    mask_array = y_true
                y_true = self.get_ytrue(mask_array, self.data)
                oor = ~(y_true == 0) * 1
                y_d = y_d * oor
                j, d, tp, tn, a, class_j = self.evaluate_one_image(y_true, y_d, labels)
                jaccard.append(j)
                dice.append(d)
                tpr.append(tp)
                tnr.append(tn)
                acc.append(a)
                print(N, "/", len(mask_path), "dataset_%d"%d_num)
                print(mask_name,j,d,tp,tn,a)
                N += 1

                with open(d_path + "image_evaluate.txt", mode = "a") as file:
                    results = mask_name+" "+str(j)+"  "+str(d)+"  "+str(tp)+"  "+str(tn)+"  "+str(a)+"\n"
                    file.write(results)

            jaccard = sum(jaccard) / len(jaccard)
            dice = sum(dice) / len(dice)
            tpr = sum(tpr) / len(tpr)
            tnr = sum(tnr) / len(tnr)
            acc = sum(acc) / len(acc)
            with open(d_path + "image_evaluate.txt", mode = "a") as file:
                    results = "All "+str(jaccard)+"  "+str(dice)+"  "+str(tpr)+"  "+str(tnr)+"  "+str(acc)+"\n"
                    file.write(results)
            print(results)

            with open(d_path + "image_evaluate.txt") as f:
                s = f.readlines()
                all_result = s[-1]
                all_result =all_result.split()
                all_j.append(float(all_result[1]))
                all_d.append(float(all_result[2]))
                all_tp.append(float(all_result[3]))
                all_tn.append(float(all_result[4]))
                all_acc.append(float(all_result[5]))
        j_a = np.average(all_j)
        d_a = np.average(all_d)
        tp_a = np.average(all_tp)
        tn_a = np.average(all_tn)
        acc_a = np.average(all_acc)
        j_s = np.std(all_j)
        d_s = np.std(all_d)
        tp_s = np.std(all_tp)
        tn_s = np.std(all_tn)
        acc_s = np.std(all_acc)
        with open("./%s/%s/%s/size_%d/seg_result_%s.txt"%(self.data,
                                                       self.method,
                                                       self.model_name,
                                                       self.patch_size,
                                                       self.stage), "w") as file:
            file.write(str(j_a) + " ± " + str(j_s) + "\n")
            file.write(str(d_a) + " ± " + str(d_s) + "\n")
            file.write(str(tp_a) + " ± " + str(tp_s) + "\n")
            file.write(str(tn_a) + " ± " + str(tn_s) + "\n")
            file.write(str(acc_a) + " ± " + str(acc_s) + "\n")
        print(self.model_name, self.patch_size)
        print("Jac.: " + str(j_a) + " ± " + str(j_s))
        print("Dice: " + str(d_a) + " ± " + str(d_s))
        print("TP  : " + str(tp_a) + " ± " + str(tp_s))
        print("TN  : " + str(tn_a) + " ± " + str(tn_s))
        print("Acc : " + str(acc_a) + " ± " + str(acc_s))
