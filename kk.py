def faceFeature(self,event):
        if(self.cap.isOpened()):
            img=self.img
            img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            
            face_pos=self.face_catch.detectMultiScale(img,1.3,5)
            x,y,w,h=face_pos[0]
            img=img[y:y+h,x:x+w]
            Y,CR,CB=cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb))
            
            CR_array,Cb_array=np.array(CR,dtype=np.int32),np.array(CB,dtype=np.int32)
            showimg("cr",CR)
            showimg("cb",CB)
            
            showbignum("CR_NOrm",CR_array)
            showbignum("Cb_array",Cb_array)
            print(CR_array,CR_array*CR_array)
            p1=1/3*(CR_array*CR_array+Cb_array*Cb_array)+(Cb_array/CR_array+0.1)
            print(p1)
            showbignum("P1",p1.astype(int))
            showbignum("cr2",CR_array*CR_array)
            showbignum("cb2",Cb_array*Cb_array)
            
            
            kernel=kernelbysize(img.shape[0])                
            Y_erode=cv2.erode(Y,kernel,iterations=1)#∏Ø ¥
            Y_dilate=cv2.dilate(Y,kernel,iterations=1)#≈Ú’Õ
            p3=p1*Y_dilate/(Y_erode+1.)
            
            p3=cv2.normalize(p3,p3,0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
            showimg("p3",p3)
            kernel_process=kernelbysize(20) 
            # p3=cv2.erode(p3,kernel_process,iterations=1)
             
            p13=cv2.dilate(p3,kernel_process,iterations=40)
            showimg("-p3",p13)
            p13=255-p13
            showimg("p13",p13)
            con,hie=cv2.findContours(p13,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            p3_copy=p13.copy()
            print(len(con)) 
            xm,ym,wm,hm= cv2.boundingRect(con[1]) 
             
            eye_pos=self.eye_catch.detectMultiScale(img)
            x0,y0,w0,h0=eye_pos[0]
            x1,y1,w1,h1=eye_pos[1]
            fin=img.copy()
            rect=cv2.rectangle(fin,(xm,ym),(xm+wm,ym+hm),(0,255,0),3)
            rect=cv2.rectangle(fin,(x0,y0),(x0+w0,y0+h0),(0,255,0),3)
            rect=cv2.rectangle(fin,(x1,y1),(x1+w1,y1+h1),(0,255,0),3)
            k=1
            if x0<xm:
                k=-1    
            first_face=(int(x0+k*0.3*w0),y0+h0,w0,ym-y0-h0)
            k=1
            if x1<xm:
                k=-1
            second_face=(int(x1+k*0.3*w1),y1+h1,w1,ym-y1-h1)
            x0,y0,w0,h0=first_face
            x1,y1,w1,h1=second_face
            rect=cv2.rectangle(fin,(x0,y0),(x0+w0,y0+h0),(0,255,0),3)
            rect=cv2.rectangle(fin,(x1,y1),(x1+w1,y1+h1),(0,255,0),3)
            mask0=np.zeros(img.shape[:2],np.uint8)

            mask0[y0:y0+h0,x0:x0+w0]=255
            mask1=np.zeros(img.shape[:2],np.uint8)
            mask1[y1:y1+h1,x1:x1+w1]=255
            fin=img.copy()
            rect=cv2.rectangle(fin,(x0,y0),(x0+w0,y0+h0),(0,255,0),3)
            rect=cv2.rectangle(fin,(x1,y1),(x1+w1,y1+h1),(0,255,0),3)
            color_df=np.zeros((3))
            for i,col in enumerate(['b','g','r']):
                hist_mask0=cv2.calcHist([img],[i],mask0,[25],[0,256])
                color_df[i]+=np.argmax(hist_mask0)
                hist_mask1=cv2.calcHist([img],[i],mask1,[25],[0,256])
                color_df[i]+=np.argmax(hist_mask1)
            color_df=color_df/2
            self.faceFeature=color_df
            self.ui.textEdit.setText(str([(k,i) for k,i in zip(['¿∂','¬Ã','¿∂'],color_df)]))