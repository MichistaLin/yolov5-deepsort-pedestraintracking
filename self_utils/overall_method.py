import cv2,os,natsort
import numpy as np


class Trace_Mask:
    def __init__(self,img_H,img_W,save_path) -> None:
        super().__init__()
        self.mask=np.zeros((img_H,img_W))
        self.save_path=save_path
        
    def update_mask(self,box):
        self.mask[box[1]:box[3],box[0]:box[2]]+=1
        return
    
    def visulize_mask(self,img):
        mask=np.ones_like(self.mask)
        mask[self.mask>0]=0
        return img
    
    def save_final_mask(self):
        cv2.imwrite(self.save_path,self.mask)
        
        
        
class Vector_Memory:
    def __init__(self,min_cosin=0.8,init_num=15,max_vector_num=4) -> None:
        super().__init__()
        self.mean_Vector=[]
        self.mean_Length=[]
        self.Num=[]
        self.min_cosin=min_cosin
        self.flag="init"
        self.init_num=init_num
        self.max_vector_num=max_vector_num

    def update(self,velocity):
        vector,length=self.standardize(velocity)
        if length>=20:
            for index,mean_vector in enumerate(self.mean_Vector):
                cosin=(np.dot(mean_vector,vector))/(np.sqrt(np.dot(mean_vector,mean_vector)*np.dot(vector,vector))+1e-4)
                if cosin>self.min_cosin:
                    self.mean_Vector[index]=np.float16((self.Num[index]*mean_vector+vector)/(self.Num[index]+1))
                    self.mean_Length[index]=np.float16((self.Num[index]*self.mean_Length[index]+length)/(self.Num[index]+1))
                    self.Num[index]+=1
                    if sum(self.Num) >= self.init_num:
                        self.flag="check"
                    return
            if len(self.Num) < self.max_vector_num:
                self.mean_Vector.append(vector)
                self.mean_Length.append(length)
                self.Num.append(1)

    def check_lawful(self,velocity):
        vector,length=self.standardize(velocity)
        if length <= 20:
            return True
        for index,mean_vector in enumerate(self.mean_Vector):
            cosin=(np.dot(mean_vector,vector))/(1e-4+np.sqrt(np.dot(mean_vector,mean_vector)*np.dot(vector,vector)))
            if cosin>self.min_cosin and (self.Num[index]>self.init_num or self.Num[index]/sum(self.Num) > (0.8/len(self.Num))):
                return True
        return False
        
    def standardize(self,velocity):
        x=np.array(velocity,dtype=np.float16)
        length=np.sqrt(np.dot(x,x))+1e-4
        x=x/length
        return x,length

class Vector_Field:
    def __init__(self,img_H=1080,img_W=1920,box_size=50) -> None:
        super().__init__()
        self.img_H=img_H
        self.img_W=img_W
        self.box_size=box_size
        self.vector_memory=[[Vector_Memory() for _ in range(1+img_W//box_size)] for _ in range(1+img_H//box_size)]

    def update(self,box,velocity):
        J,I=(int((box[0]+box[2])/2)//self.box_size,int((box[1]+box[3])/2)//self.box_size)
        Box=[]
        for ii in range(3):
            for jj in range(3):
                try:
                    if self.vector_memory[I-1+ii][J-1+jj].flag=="init":
                        self.vector_memory[I-1+ii][J-1+jj].update(velocity)
                        Box.append(1)
                    else:
                        self.vector_memory[I-1+ii][J-1+jj].update(velocity)
                        if self.vector_memory[I-1+ii][J-1+jj].check_lawful(velocity):
                            Box.append(1)
                        else:
                            Box.append(0)
                except:
                    pass
        return sum(Box) >= 0.6*len(Box)

    def draw_vector_field(self,img=None):
        if img is None or img.shape[0]!=self.img_H or img.shape[1]!=self.img_W:
            img=np.ones((self.img_H,self.img_W,3))
            
        for I in range(len(self.vector_memory)):
            for J in range(len(self.vector_memory[I])):
                box_center=(J*self.box_size+int(self.box_size/2),I*self.box_size+int(self.box_size/2))
                if sum(self.vector_memory[I][J].Num) == 0:
                    cv2.circle(img, box_center, 2, (0,170,170), 2)
                else:
                    cv2.circle(img, box_center, 2, (0,240,0), 2)
                    for index,vector in enumerate(self.vector_memory[I][J].mean_Vector):
                        if (self.vector_memory[I][J].Num[index]>self.vector_memory[I][J].init_num or self.vector_memory[I][J].Num[index]/sum(self.vector_memory[I][J].Num) > (0.8/len(self.vector_memory[I][J].Num))):
                            pointat=(box_center[0]+int(20*vector[0]),box_center[1]+int(20*vector[1]))
                            cv2.arrowedLine(img,box_center, pointat, (0,240,0),2,0,0,0.3)
        return img


class Object_Counter:
    def __init__(self,name_list) -> None:
        super().__init__()

    def draw_counter(self, img, present_num, total_num, text, isCountPresent, color=[255, 0, 0], thickness=None, fontsize=None):
        thickness = max(2, round(0.0016 * (img.shape[0] + img.shape[1]) / 2)) if thickness == None else thickness
        fontsize = 0.5 * thickness if fontsize == None else fontsize
        top = (5, 5)
        if isCountPresent:
            text_info = "{}: {}".format(text, present_num)
        else:
            text_info = "{}: {}".format(text, total_num)
        t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize, thickness + 2)[0]
        cv2.putText(img, text_info, (top[0], top[1] + t_size[1] + 2), cv2.FONT_HERSHEY_TRIPLEX, fontsize, color,
                    thickness)
        if not isCountPresent:
            text_info1 = "{}: {}".format("present person", present_num)
            cv2.putText(img, text_info1, (top[0], top[1] + t_size[1] + 2 + 30), cv2.FONT_HERSHEY_TRIPLEX, fontsize,
                        color,
                        thickness)

        return img

        
class Image_Capture:
    def __init__(self,source) -> None:
        super().__init__()
        self.source=os.path.dirname(source) if (source.endswith(".jpg") or source.endswith(".png")) else source
        if os.path.isdir(self.source):
            self.source_type="imgs" 
        elif source.isdigit():
            self.source_type="camera"
            self.source=int(self.source)
        elif source.startswith("rtsp") or source.startswith("rtmp"):
            self.source_type="camera"
        else:
            self.source_type="video"
        self.index=0
        self.ret=True

        if self.source_type == "imgs":
            if (source.endswith(".jpg") or source.endswith(".png")):
                self.img_List=[os.path.basename(source)]
            else:
                self.img_List=natsort.natsorted(os.listdir(source))
            _,img,_=self.read()
            self.index-=1
            self.shape=img.shape
        else:
            self.cap=cv2.VideoCapture(self.source)
        
    def read(self):
        if self.source_type == "imgs":
            img=cv2.imread(os.path.join(self.source,self.img_List[self.index]))
            ret = True if hasattr(img, 'shape') else False
            self.index+=1
            self.ret=ret
            return ret,img,self.img_List[self.index-1]
        elif self.source_type == "camera":
            ret,img=self.cap.read()
            self.index+=1
            self.ret=ret
            return ret,img,"frame_{}.jpg".format(self.index)
        else:
            ret,img=self.cap.read()
            self.ret=ret
            return ret,img
            
    def get(self,i=0):
        if self.source_type == "imgs":
            if i==1:
                return self.index
            if i==7:
                return len(self.img_List)
            if i==4:
                return self.shape[0]
            if i==3:
                return self.shape[1]
            
        elif self.source_type == "camera":
            return self.index if i==1 else int(self.cap.get(i))
        
        else:
            return int(self.cap.get(i))
    
    def get_index(self):
        return self.get(1)
    
    def get_length(self):
        return self.get(7)
    
    def get_height(self):
        return self.get(4)
    
    def get_width(self):
        return self.get(3)
    
    def ifcontinue(self):
        if self.source_type == "imgs":
            return (self.index < len(self.img_List)) and self.ret
        else:
            return (self.cap.get(1) < self.cap.get(7) or self.cap.get(7) <= 0) and self.ret

    def release(self):
        if self.source_type == "imgs":
            pass
        else:
            self.cap.release()
