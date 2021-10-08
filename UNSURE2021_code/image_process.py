import cv2
import numpy as np
from PIL import Image
from operator import itemgetter



class PatchImageLoader(object):

    def __init__(self,
                 img_path,
                 patch_size,
                 train_slide_step,
                 data,
                 mode):
        img_path.sort()
        self.img_path = img_path
        self.patch_size = patch_size
        self.train_slide_step = train_slide_step
        self.data = data
        self.mode = mode

    def count_samples(self):
        samples_num = 0
        for i_path in self.img_path:
            img = Image.open(i_path)
            width, height = img.size
            for y in range(0, height-self.patch_size+1, self.train_slide_step):
                for x in range(0, width-self.patch_size+1, self.train_slide_step):
                    samples_num += 1
        return samples_num

    def crop_img(self):
        img_list = []
        for i_path in self.img_path:
            img = Image.open(i_path)
            width, height = img.size
            for y in range(0, height-self.patch_size+1, self.train_slide_step):
                for x in range(0, width-self.patch_size+1, self.train_slide_step):
                    crop_img = np.array(img.crop((x, y,
                                                  x + self.patch_size,
                                                  y + self.patch_size)))
                    crop_img.resize(self.patch_size, self.patch_size, 3)
                    img_list.append(crop_img)
        return img_list

    def crop_one_img(self, i_path, ref_num, reflect=False):
        img_list = []
        img = Image.open(i_path)
        width, height = img.size
        if reflect:
            height = original_height + ref_num*2
            width = original_width + ref_num*2
            img = np.array(img)
            img = cv2.copyMakeBorder(img, self.patch_size, self.patch_size, self.patch_size, self.patch_size,cv2.BORDER_REFLECT)
            img = Image.fromarray(np.uint8(img))
        for y in range(0, height-self.patch_size+1, self.train_slide_step):
            for x in range(0, width-self.patch_size+1, self.train_slide_step):
                crop_img = np.array(img.crop((x, y,
                                                x + self.patch_size,
                                                y + self.patch_size)))
                crop_img.resize(self.patch_size, self.patch_size, 3)
                img_list.append(crop_img)
        return img_list, width, height#original_width, original_height

class PatchLabelLoader(object):

    def __init__(self,
                 mask_path,
                 patch_size,
                 train_slide_step,
                 data,
                 mode):
        mask_path.sort()
        self.mask_path = mask_path
        self.patch_size = patch_size
        self.train_slide_step = train_slide_step
        self.data = data
        self.mode = mode

    def crop_resolution_mask(self, res_lv):
        mask_list = []
        for m_path in self.mask_path:
            mask_image = cv2.imread(m_path)
            height, width = mask_image.shape[:2]
            for y in range(0, height-self.patch_size+1, self.train_slide_step):
                for x in range(0, width-self.patch_size+1, self.train_slide_step):
                    crop_mask = mask_image[y:self.patch_size + y,
                                           x:self.patch_size + x]
                    hist = self.rotation_get_hist(crop_mask, res_lv)
                    mask_list.append(hist)
        return mask_list

    def crop_melanoma_mask(self):
        mask_list = []
        for m_path in self.mask_path:
            mask_image = Image.open(m_path)
            mask_array = np.array(mask_image)
            height, width = mask_array.shape
            label = (mask_array==0) * np.ones((height, width))
            mask = Image.fromarray(np.uint8(label))
            for y in range(0, height-self.patch_size+1, self.train_slide_step):
                for x in range(0, width-self.patch_size+1, self.train_slide_step):
                    m_label = np.array(mask.crop((x, y, x + self.patch_size, y + self.patch_size)))
                    mask_list.append(m_label)
        return mask_list

    def crop_ips_mask(self):
        mask_list = []
        for m_path in self.mask_path:
            mask_image = Image.open(m_path)
            mask_array = np.array(mask_image)[:,:,:3]
            height, width = mask_array.shape[:2]
            good_label = ((mask_array[:,:,0]==255)&(mask_array[:,:,1]==0)&(mask_array[:,:,2]==0))*np.ones((height, width))
            bad_label = ((mask_array[:,:,0]==0)&(mask_array[:,:,1]==255)&(mask_array[:,:,2]==0))*np.ones((height, width))*2
            bgd_label = ((mask_array[:,:,0]==0)&(mask_array[:,:,1]==0)&(mask_array[:,:,2]==255))*np.ones((height, width))*3
            mask_label = good_label + bad_label + bgd_label
            mask = Image.fromarray(np.uint8(mask_label))
            for y in range(0, height-self.patch_size+1, self.train_slide_step):
                for x in range(0, width-self.patch_size+1, self.train_slide_step):
                    m_label = np.array(mask.crop((x, y, x + self.patch_size, y + self.patch_size)))
                    mask_list.append(m_label)
        return mask_list

    def cv2_mask(self):
        pre_mask_list = []
        for m_path in self.mask_path:
            im2 = cv2.imread(m_path)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
            height, width = im2.shape[:2]
            for y in range(0, height-self.patch_size+1, self.train_slide_step):
                for x in range(0, width-self.patch_size+1, self.train_slide_step):
                    c_im2 = im2[y:self.patch_size+y, x:self.patch_size+x]
                    #c_im2.resize(473,473,3)
                    imim = c_im2[:,:,0]+c_im2[:,:,1]+c_im2[:,:,2]
                    c_im2[:,:,0]=(imim==255)*c_im2[:,:,0]
                    c_im2[:,:,1]=(imim==255)*c_im2[:,:,1]
                    c_im2[:,:,2]=(imim==255)*c_im2[:,:,2]
                    pre_mask_list.append(c_im2)
        return pre_mask_list

    def rotation_get_hist(self, crop_mask, res_lv):
        """
        size: int
        res_lv: int
        data: str, "ips" or "melanoma"
        hist_list
        Fix: lとかx,yのloopのstartpointの名前をわかりやすくする。
        """
        hist_list = []
        res_list = []
        r_size = int(self.patch_size / 4)
        for r in range(res_lv):
            tmp = r_size
            res_list.append(r_size)
            r_size = int(r_size/2)
        res_list.append(tmp)
        l = 0
        for i in res_list:
            x = 0
            y = 0
            for x in range(l, self.patch_size-l-i+1, i):
                sub_mask = crop_mask[y:i+y, x:i+x]
                hist = self.get_pmf(sub_mask)
                hist_list.append(hist)
            for y in range(l+i, self.patch_size-l-i+1, i):
                sub_mask = crop_mask[y:i+y, x:i+x]
                hist = self.get_pmf(sub_mask)
                hist_list.append(hist)
            for x in range(x-i, l-1, -i):
                sub_mask = crop_mask[y:i+y, x:i+x]
                hist = self.get_pmf(sub_mask)
                hist_list.append(hist)
            for y in range(y-i, l, -i):
                sub_mask = crop_mask[y:i+y, x:i+x]
                hist = self.get_pmf(sub_mask)
                hist_list.append(hist)
            l += i
        hist_list = np.asarray(hist_list)
        return hist_list

    def get_pmf(self, sub_mask):
        size = sub_mask.shape
        size = size[0]
        sub_mask = sub_mask // 255
        if self.data == "ips":
            # ips dataset
            # good -> 0, bad -> 1, bgd -> 2, others -> 3　とする
            img_label = (sub_mask[:, :, 0] * 1) + \
                        (sub_mask[:, :, 1] * 2) + \
                        (sub_mask[:, :, 2] * 3)
            # よくわからないラベルを全てothers へ screening
            img_label[img_label == 0] = 4
            img_label[img_label > 3] = 4
            img_label[img_label == 4] = 0
            good = (img_label == 1).sum()
            bad = (img_label == 2).sum()
            bgd = (img_label == 3).sum()
            others = (img_label == 0).sum()
            sum_la = good+bad+bgd
            if sum_la * 0.2 < others:
                return [0, 0, 0]
            hist = [good, bad, bgd]
            if np.min(hist) < 100:
                label = np.argmin(hist) + 1
                img_label[img_label == label] = 0
            else:
                img_label = img_label - 1
            return hist/sum_la

        # binary mask
        else:
            img_label = sub_mask
            target = (img_label == 1).sum()
            bgd = (img_label == 0).sum()
            sum_label = target + bgd
            hist = [target, bgd]
            if np.min(hist) < 100:
                label = np.argmin(hist) + 1
                img_label[img_label == label] = 0
            return hist/sum_label

class PatchSRLoader(object):

    def __init__(self,
                 img_path,
                 mask_path,
                 pre_path,
                 patch_size,
                 step,
                 data,
                 mode):
        img_path.sort()
        self.img_path = img_path
        mask_path.sort()
        self.mask_path = mask_path
        pre_path.sort()
        self.pre_path = pre_path
        self.patch_size = patch_size
        self.step = step
        self.data = data
        self.mode = mode
        label = [1,2,3]

    def count_samples(self):
        samples_num = 0

        for m_path,p_path in zip(self.mask_path,self.pre_path):
            y=0
            u=0
            d=0
            m_mask = cv2.imread(m_path)
            m_mask = cv2.cvtColor(m_mask, cv2.COLOR_BGR2RGB)
            height, width = m_mask.shape[:2]

            p_image = cv2.imread(p_path)
            p_image = cv2.cvtColor(p_image, cv2.COLOR_BGR2RGB)

            while y < (height - self.patch_size + 1):
                x = 0
                while x < (width - self.patch_size + 1):
                    crop_mask = m_mask[y:self.patch_size+y, x:self.patch_size+x]
                    crop_mt = self.get_ytrue(crop_mask, 'ips')
                    oor = ~(crop_mt == 0) *1

                    crop_p = p_image[y:self.patch_size+y, x:self.patch_size+x]
                    crop_pt = self.get_ytrue(crop_p, 'ips')
                    crop_pt = crop_pt * oor

                    #y_pred = self.get_ytrue(crop_p, 'ips')
                    acc=round(np.count_nonzero((crop_mt-crop_pt)==0)/(self.patch_size*self.patch_size),3)*10

                    #a=int((((self.patch_size/10-1)/3**5)*3**(acc-5)+1)*10)
                    x+=self.step#self.patch_size//10
                    if acc >= 8:
                        u+=1
                    else:
                        d+=1
                y+=self.step#self.patch_size//10
            if u==0 :
                samples_num+=d
            elif d == 0:
                samples_num+=3
            #else:
                #samples_num+=d
            elif u<=d:
                samples_num+=(d+u)
            else:
                samples_num+=(d+d)
        return samples_num

    def input(self):
        inputupA=[]
        inputupB=[]
        trueup=[]
        tup=[]
        inputdownA=[]
        inputdownB=[]
        truedown=[]
        tdown=[]
        A=[]
        B=[]
        C=[]
        D=[]
        for i_path, m_path,p_path in zip(self.img_path, self.mask_path,self.pre_path):
            y=0
            m_mask = cv2.imread(m_path)
            m_mask = cv2.cvtColor(m_mask, cv2.COLOR_BGR2RGB)
            height, width = m_mask.shape[:2]
            imim = m_mask[:,:,0]+m_mask[:,:,1]+m_mask[:,:,2]
            m_mask[:,:,0]=(imim==255)*m_mask[:,:,0]
            m_mask[:,:,1]=(imim==255)*m_mask[:,:,1]
            m_mask[:,:,2]=(imim==255)*m_mask[:,:,2]
            

            p_image = cv2.imread(p_path)
            p_image = cv2.cvtColor(p_image, cv2.COLOR_BGR2RGB)
            #p_g = cv2.cvtColor(p_mask, cv2.COLOR_BGR2GRAY)
            img = Image.open(i_path).convert('RGB')
            img = np.array(img)
            while y < (height - self.patch_size + 1):
                x = 0
                while x < (width - self.patch_size + 1):
                    crop_mask = m_mask[y:self.patch_size+y, x:self.patch_size+x]
                    cm = crop_mask.copy()
                    cm.resize(self.patch_size//16,self.patch_size//16,3)
                    crop_mt = self.get_ytrue(crop_mask, 'ips')
                    oor = ~(crop_mt == 0) *1

                    crop_p = p_image[y:self.patch_size+y, x:self.patch_size+x]
                    crop_pt = self.get_ytrue(crop_p, 'ips')
                    crop_pt = crop_pt * oor#

                    crop_img = img[y:self.patch_size+y, x:self.patch_size+x]
                    acc=round(np.count_nonzero((crop_mt-crop_pt)==0)/(self.patch_size*self.patch_size),3)*10

                    #a=int((((self.patch_size/10-1)/3**5)*3**(acc-5)+1)*10)
                    x+=self.step
                    if acc >= 8 :
                        #inputup.append(np.concatenate([crop_p,crop_img],axis=2))
                        inputupA.append(crop_p)
                        inputupB.append(crop_img)
                        trueup.append(crop_mask)
                        tup.append(cm)
                    else:
                        #inputdown.append(np.concatenate([crop_p,crop_img],axis=2))
                        inputdownA.append(crop_p)
                        inputdownB.append(crop_img)
                        truedown.append(crop_mask)
                        tdown.append(cm)
                y+=self.step
        if len(inputdownA)==0 or len(inputdownA)==1:
            shuffled_idx = np.random.choice(np.arange(len(inputupA)), 3, replace=True)
            A.extend(itemgetter(*shuffled_idx)(inputupA))
            B.extend(itemgetter(*shuffled_idx)(inputupB))
            C.extend(itemgetter(*shuffled_idx)(trueup))
            D.extend(itemgetter(*shuffled_idx)(tup))
            return A,B,C,D
        elif len(inputupA)==0:
            #shuffled_idx = np.random.choice(np.arange(len(inputdown)), 3, replace=True)
            return inputdownA,inputdownB,truedown,tdown#itemgetter(*shuffled_idx)(inputdown),itemgetter(*shuffled_idx)(truedown)
        elif len(inputupA)!=0 and len(inputupA) <= len(inputdownA):
            #shuffled_idx = np.random.choice(np.arange(len(inputdown)), len(inputup), replace=True)
            #inputup.extend(itemgetter(*shuffled_idx)(inputdown))
            #trueup.extend(itemgetter(*shuffled_idx)(truedown))
            inputupA.extend(inputdownA)
            inputupB.extend(inputdownB)
            trueup.extend(truedown)
            tup.extend(tdown)
            return inputupA,inputupB,trueup,tup
        else:
            shuffled_idx = np.random.choice(np.arange(len(inputupA)), len(inputdownA), replace=True)
            inputdownA.extend(itemgetter(*shuffled_idx)(inputupA))
            inputdownB.extend(itemgetter(*shuffled_idx)(inputupB))
            truedown.extend(itemgetter(*shuffled_idx)(trueup))
            tdown.extend(itemgetter(*shuffled_idx)(tup))
            return inputdownA,inputdownB,truedown,tdown

    def testinput(self, img_path,mask_path,pre_path):
        c_i=[]
        c_h=[]
        c_t=[]
        y=0
        p_image = cv2.imread(pre_path)
        p_image = cv2.cvtColor(p_image, cv2.COLOR_BGR2RGB)
        height, width = p_image.shape[:2]
        #m_mask = cv2.imread(mask_path)
        #m_mask = cv2.cvtColor(m_mask, cv2.COLOR_BGR2RGB)
        n=0
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        while y < (height - self.patch_size + 1):
            x = 0
            while x < (width - self.patch_size + 1):
                crop_p = p_image[y:self.patch_size+y, x:self.patch_size+x]/255
                crop_img = img[y:self.patch_size+y, x:self.patch_size+x]/255.
                #crop_mask = m_mask[y:self.patch_size+y, x:self.patch_size+x]
                #c_t.append(np.concatenate([crop_p,crop_img],axis=2))
                c_i.append(crop_p)
                c_h.append(crop_img)
                #c_t.append(crop_mask)
                x+=self.step
            y+=self.step
        return np.array(c_i),np.array(c_h)
 
    def Newinput(self, res_lv):
        inputupA=[]
        inputupB=[]
        trueup=[]
        #tup=[]
        inputdownA=[]
        inputdownB=[]
        truedown=[]
        #tdown=[]
        A=[]
        B=[]
        C=[]
        #D=[]
        for i_path, m_path,p_path in zip(self.img_path, self.mask_path,self.pre_path):
            y=0
            m_mask = cv2.imread(m_path)
            m_mask = cv2.cvtColor(m_mask, cv2.COLOR_BGR2RGB)
            height, width = m_mask.shape[:2]
            imim = m_mask[:,:,0]+m_mask[:,:,1]+m_mask[:,:,2]
            m_mask[:,:,0]=(imim==255)*m_mask[:,:,0]
            m_mask[:,:,1]=(imim==255)*m_mask[:,:,1]
            m_mask[:,:,2]=(imim==255)*m_mask[:,:,2]
            

            p_image = cv2.imread(p_path)
            p_image = cv2.cvtColor(p_image, cv2.COLOR_BGR2RGB)
            #p_g = cv2.cvtColor(p_mask, cv2.COLOR_BGR2GRAY)
            img = Image.open(i_path).convert('RGB')
            img = np.array(img)
            while y < (height - self.patch_size + 1):
                x = 0
                while x < (width - self.patch_size + 1):
                    crop_mask = m_mask[y:self.patch_size+y, x:self.patch_size+x]
                    hist = self.rotation_get_hist(crop_mask, res_lv)
                    #cm = crop_mask.copy()
                    #cm.resize(self.patch_size//16,self.patch_size//16,3)
                    crop_mt = self.get_ytrue(crop_mask, 'ips')
                    oor = ~(crop_mt == 0) *1

                    crop_p = p_image[y:self.patch_size+y, x:self.patch_size+x]
                    crop_pt = self.get_ytrue(crop_p, 'ips')
                    crop_pt = crop_pt * oor#

                    crop_img = img[y:self.patch_size+y, x:self.patch_size+x]
                    acc=round(np.count_nonzero((crop_mt-crop_pt)==0)/(self.patch_size*self.patch_size),3)*10

                    #a=int((((self.patch_size/10-1)/3**5)*3**(acc-5)+1)*10)
                    x+=self.step
                    if acc >= 8 :
                        #inputup.append(np.concatenate([crop_p,crop_img],axis=2))
                        inputupA.append(crop_p)
                        inputupB.append(crop_img)
                        trueup.append(hist)
                        #tup.append(cm)
                    else:
                        #inputdown.append(np.concatenate([crop_p,crop_img],axis=2))
                        inputdownA.append(crop_p)
                        inputdownB.append(crop_img)
                        truedown.append(hist)
                        #tdown.append(cm)
                y+=self.step
        if len(inputdownA)==0 or len(inputdownA)==1:
            shuffled_idx = np.random.choice(np.arange(len(inputupA)), 3, replace=True)
            A.extend(itemgetter(*shuffled_idx)(inputupA))
            B.extend(itemgetter(*shuffled_idx)(inputupB))
            C.extend(itemgetter(*shuffled_idx)(trueup))
            #D.extend(itemgetter(*shuffled_idx)(tup))
            return A,B,C#,D
        elif len(inputupA)==0:
            #shuffled_idx = np.random.choice(np.arange(len(inputdown)), 3, replace=True)
            return inputdownA,inputdownB,truedown#,tdown #itemgetter(*shuffled_idx)(inputdown),itemgetter(*shuffled_idx)(truedown)
        elif len(inputupA)!=0 and len(inputupA) <= len(inputdownA):
            #shuffled_idx = np.random.choice(np.arange(len(inputdown)), len(inputup), replace=True)
            #inputup.extend(itemgetter(*shuffled_idx)(inputdown))
            #trueup.extend(itemgetter(*shuffled_idx)(truedown))
            inputupA.extend(inputdownA)
            inputupB.extend(inputdownB)
            trueup.extend(truedown)
            #tup.extend(tdown)
            return inputupA,inputupB,trueup#,tup
        else:
            shuffled_idx = np.random.choice(np.arange(len(inputupA)), len(inputdownA), replace=True)
            inputdownA.extend(itemgetter(*shuffled_idx)(inputupA))
            inputdownB.extend(itemgetter(*shuffled_idx)(inputupB))
            truedown.extend(itemgetter(*shuffled_idx)(trueup))
            #tdown.extend(itemgetter(*shuffled_idx)(tup))
            return inputdownA,inputdownB,truedown#,tdown

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

    def evaluate_one_image(self, y_true, y_pred, labels):
        """
        evaluate segmentation result, using confusion_matrix
        y_true: 2d array,
        y_pred: 2d array,
        labels: the labels dataset contains,
                ips -> [1, 2, 3], melanoma -> [0, 1]
        oor is ignored.
        """
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        mat = confusion_matrix(y_true, y_pred, labels=labels)
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

    def rotation_get_hist(self, crop_mask, res_lv):
        """
        size: int
        res_lv: int
        data: str, "ips" or "melanoma"
        hist_list
        Fix: lとかx,yのloopのstartpointの名前をわかりやすくする。
        """
        hist_list = []
        res_list = []
        r_size = int(self.patch_size / 4)
        for r in range(res_lv):
            tmp = r_size
            res_list.append(r_size)
            r_size = int(r_size/2)
        res_list.append(tmp)
        l = 0
        for i in res_list:
            x = 0
            y = 0
            for x in range(l, self.patch_size-l-i+1, i):
                sub_mask = crop_mask[y:i+y, x:i+x]
                hist = self.get_pmf(sub_mask)
                hist_list.append(hist)
            for y in range(l+i, self.patch_size-l-i+1, i):
                sub_mask = crop_mask[y:i+y, x:i+x]
                hist = self.get_pmf(sub_mask)
                hist_list.append(hist)
            for x in range(x-i, l-1, -i):
                sub_mask = crop_mask[y:i+y, x:i+x]
                hist = self.get_pmf(sub_mask)
                hist_list.append(hist)
            for y in range(y-i, l, -i):
                sub_mask = crop_mask[y:i+y, x:i+x]
                hist = self.get_pmf(sub_mask)
                hist_list.append(hist)
            l += i
        hist_list = np.asarray(hist_list)
        return hist_list

    def get_pmf(self, sub_mask):
        size = sub_mask.shape
        size = size[0]
        sub_mask = sub_mask // 255
        if self.data == "ips":
            # ips dataset
            # good -> 0, bad -> 1, bgd -> 2, others -> 3　とする
            img_label = (sub_mask[:, :, 0] * 1) + \
                        (sub_mask[:, :, 1] * 2) + \
                        (sub_mask[:, :, 2] * 3)
            # よくわからないラベルを全てothers へ screening
            img_label[img_label == 0] = 4
            img_label[img_label > 3] = 4
            img_label[img_label == 4] = 0
            good = (img_label == 1).sum()
            bad = (img_label == 2).sum()
            bgd = (img_label == 3).sum()
            others = (img_label == 0).sum()
            sum_la = good+bad+bgd
            if sum_la * 0.2 < others:
                return [0, 0, 0]
            hist = [good, bad, bgd]
            if np.min(hist) < 100:
                label = np.argmin(hist) + 1
                img_label[img_label == label] = 0
            else:
                img_label = img_label - 1
            return hist/sum_la

        # binary mask
        else:
            img_label = sub_mask
            target = (img_label == 1).sum()
            bgd = (img_label == 0).sum()
            sum_label = target + bgd
            hist = [target, bgd]
            if np.min(hist) < 100:
                label = np.argmin(hist) + 1
                img_label[img_label == label] = 0
            return hist/sum_label
