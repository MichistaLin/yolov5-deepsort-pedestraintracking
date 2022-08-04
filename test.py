import numpy as np
import random,math,cv2
        
class vector_memory:
    def __init__(self,min_cosin=0.866,init_num=20,lawful_threshold=0.2) -> None:
        super().__init__()
        self.mean_Vector=[]
        self.mean_Length=[]
        self.Num=[]
        self.min_cosin=min_cosin
        self.flag="init"
        self.init_num=init_num
        self.lawful_threshold=lawful_threshold
        
    def update(self,velocity):
        vector,length=self.standardize(velocity)
        for index,mean_vector in enumerate(self.mean_Vector):
            cosin=(np.dot(mean_vector,vector))/(np.sqrt(np.dot(mean_vector,mean_vector)*np.dot(vector,vector)))
            if cosin>self.min_cosin:
                self.mean_Vector[index]=np.float16((self.Num[index]*mean_vector+vector)/(self.Num[index]+1))
                self.mean_Length[index]=np.float16((self.Num[index]*self.mean_Length[index]+length)/(self.Num[index]+1))
                self.Num[index]+=1
                if sum(self.Num) >= self.init_num:
                    self.flag="check"
                return
        self.mean_Vector.append(vector)
        self.mean_Length.append(length)
        self.Num.append(1)

    def check_lawful(self,velocity):
        vector,length=self.standardize(velocity)
        for index,mean_vector in enumerate(self.mean_Vector):
            cosin=(np.dot(mean_vector,vector))/(np.sqrt(np.dot(mean_vector,mean_vector)*np.dot(vector,vector)))
            if cosin>self.min_cosin and self.Num[index]/sum(self.Num) > self.lawful_threshold:
                return True
        return False
        
    def standardize(self,velocity):
        x=np.array(velocity,dtype=np.float16)
        length=np.sqrt(np.dot(x,x))
        x=x/length
        return x,length


class img_vector_field:
    def __init__(self,img_H=1080,img_W=1920,box_size=50) -> None:
        super().__init__()
        self.img_H=img_H
        self.img_W=img_W
        self.box_size=box_size
        self.vector_memory=[[vector_memory() for _ in range(img_W//box_size)] for _ in range(img_H//box_size)]
        
    def draw_vector_field(self,img=None):
        if img is None or img.shape[0]!=self.img_H or img.shape[1]!=self.img_W:
            img=np.ones((self.img_H,self.img_W,3))
            
        for I in range(len(self.vector_memory)):
            for J in range(len(self.vector_memory[I])):
                box_center=(J*self.box_size+int(self.box_size/2),I*self.box_size+int(self.box_size/2))
                print(I,J,box_center)
                if sum(self.vector_memory[I][J].Num) <= self.vector_memory[I][J].init_num :
                    cv2.circle(img, box_center, 2, (0,170,170), 2)
                else:
                    cv2.circle(img, box_center, 2, (0,240,0), 2)
                    for index,vector in enumerate(self.vector_memory[I][J].mean_Vector):
                        if self.vector_memory[I][J].Num[index]/sum(self.vector_memory[I][J].Num) >= self.vector_memory[I][J].lawful_threshold:
                            pointat=(box_center[1]+int(20*vector[1]),box_center[0]+int(20*vector[0]))
                            cv2.arrowedLine(img,box_center, pointat, (0,240,0),2,0,0,0.3)
        return img
                
                
            
if __name__=="__main__":
    X=[[-2,0] for _ in range(30)]
    Feild=img_vector_field()
    for x in X:
        Feild.vector_memory[0][0].update(x)
        
    img=cv2.imread("0.jpg")
    result=Feild.draw_vector_field(img)
    cv2.imwrite("1.jpg",result)