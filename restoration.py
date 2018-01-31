import cv2
import numpy as np

img1 = cv2.imread('input1.bmp')
img2 = cv2.imread('input2.bmp')
img3 = cv2.imread('input3.bmp')
img1_ori = cv2.imread('input1_ori.bmp')
img2_ori = cv2.imread('input2_ori.bmp')
img3_ori = cv2.imread('input3_ori.bmp')

def color_img_fft(img):
	img_freq = np.empty((img.shape[0],img.shape[1],img.shape[2]),dtype=np.complex128)
	img_freq[:,:,0] = np.fft.fft2(img[:,:,0])
	img_freq[:,:,1] = np.fft.fft2(img[:,:,1])
	img_freq[:,:,2] = np.fft.fft2(img[:,:,2])
	return img_freq

def color_img_ifft(img_freq):
	img = np.empty((img_freq.shape[0],img_freq.shape[1],img_freq.shape[2]),dtype=np.complex128)
	img[:,:,0] = np.fft.ifft2(img_freq[:,:,0])
	img[:,:,1] = np.fft.ifft2(img_freq[:,:,1])
	img[:,:,2] = np.fft.ifft2(img_freq[:,:,2])
	return img

def gauss2D(shape=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gauss_pad(g_f, img_shape):
    x_gf,y_gf = g_f.shape
    x1 = x_gf/2+1
    x2 = x_gf/2
    y1 = y_gf/2+1
    y2 = y_gf/2
    gf_pad = np.zeros((img_shape[0],img_shape[1]),dtype=g_f.dtype)
    print x1,x2,y1,y2
    gf_pad[:x1,:y1] = g_f[x2:,y2:]		#D
    gf_pad[-x2:,:y1] = g_f[:x2,y2:]     #C
    gf_pad[:x1,-y2:] = g_f[x2:,:y2]     #B
    gf_pad[-x2:,-y2:] = g_f[:x2,:y2]    #A
	
    return gf_pad

def weiner_filter(K,H,G):
    F = G
    F[:,:,0] = G[:,:,0]*(np.conj(H))/(H*np.conj(H)+K)
    F[:,:,1] = G[:,:,1]*(np.conj(H))/(H*np.conj(H)+K)
    F[:,:,2] = G[:,:,2]*(np.conj(H))/(H*np.conj(H)+K)
    return F

def wiener_w_contrain(r,H,G,P):
    F = G
    F[:,:,0] = G[:,:,0]*(np.conj(H))/(H*np.conj(H)+r*P*np.conj(P))
    F[:,:,1] = G[:,:,1]*(np.conj(H))/(H*np.conj(H)+r*P*np.conj(P))
    F[:,:,2] = G[:,:,2]*(np.conj(H))/(H*np.conj(H)+r*P*np.conj(P))
    return F

    
def FFT(img):

    m = cv2.getOptimalDFTSize( img.shape[0] );
    n = cv2.getOptimalDFTSize( img.shape[1] ); #on the border add zero values

    padded = cv2.copyMakeBorder(img, 0, m - img.shape[0], 0, n - img.shape[1], cv2.BORDER_REPLICATE);

    return color_img_fft(padded)
"""
def motion_blur(img_shape,a,b,T):
    H = np.empty((img_shape[0],img_shape[1]),dtype=np.complex128)
    for u in range(img_shape[0]):
        for v in range(img_shape[1]):
            H[u,v] = T/(np.pi*(u*a+v*b))*np.sin(np.pi*(u*a+v*b))*np.exp((-j)*(np.pi*(u*a+v*b)))
    return H
"""
def mod_motion_process(len, rotate):
    f = np.zeros((len,len))
    for i in range(len):
        #print len
        f[len/2,i]=(i+1)*(i+1)
    print f
    M = cv2.getRotationMatrix2D((len/2,len/2),rotate,1)
    dst = cv2.warpAffine(f,M,(len,len))
    #print dst
    #print np.sum(dst)
    dst = dst/np.sum(dst)
    #print dst
    return dst


def motion_process(len, rotate):
    f = np.zeros((len,len))
    f[len/2,:]=1
    print f
    M = cv2.getRotationMatrix2D((len/2,len/2),rotate,1)
    dst = cv2.warpAffine(f,M,(len,len))
    #print dst
    #print np.sum(dst)
    dst = dst/np.sum(dst)
    #print dst
    return dst

def PSNR(out,ori):
    MSE_B = np.sum(np.abs(out[:,:,0]-ori[:,:,0])**2)/(out.shape[0]*out.shape[1])
    MSE_G = np.sum(np.abs(out[:,:,1]-ori[:,:,1])**2)/(out.shape[0]*out.shape[1])
    MSE_R = np.sum(np.abs(out[:,:,2]-ori[:,:,2])**2)/(out.shape[0]*out.shape[1])
    PSNR_B = 10*np.log10(255*255/MSE_B)
    PSNR_G = 10*np.log10(255*255/MSE_G)
    PSNR_R = 10*np.log10(255*255/MSE_R)
    return PSNR_B+PSNR_G+PSNR_R
    
P1 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
P2 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
P3 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])


#processing img1

#trying zero pad for boundary problem but fail
#img1_freq = FFT(img1)
#img1_pad = np.empty((img.shape[0],img.shape[1],img.shape[2]))
#img1_pad = np.pad(img1, 0, mode='constant')
#img1_pad = np.zeros((img1.shape[0]+2,img1.shape[1]+2,img1.shape[2]))
#for channel in range(3):
#    img1_pad[:,:,channel] = np.pad(img1[:,:,channel], 1, mode='constant')

#img1 FFT
img1_freq = color_img_fft(img1)
img1_ori_freq = color_img_fft(img1_ori)

P1_pad = gauss_pad(P1,img1_freq.shape)
P1_freq = np.fft.fft2(P1_pad)

#kernel zero pad and FFT
gauss_filter_1d = cv2.getGaussianKernel(ksize=41,sigma=9)
gauss_filter = np.outer(gauss_filter_1d,gauss_filter_1d)
gauss_filter_pad = gauss_pad(gauss_filter,img1_freq.shape)
gauss_freq = np.fft.fft2(gauss_filter_pad)

#deconvolve and ifft
denoise_img1_freq = wiener_w_contrain(14,gauss_freq,img1_freq,P1_freq)
#denoise_img1_freq = weiner_filter(0.1,gauss_freq,img1_freq)
denoise_img1 = color_img_ifft(denoise_img1_freq)
denoise_img1 = np.abs(denoise_img1)
denoise_test = denoise_img1
denoise_img1 = np.where(denoise_img1>255,255,denoise_img1)
denoise_img1 = np.where(denoise_img1<0,0,denoise_img1)
denoise_img1 = denoise_img1.astype(np.uint8)

#trying zero pad for boundary problem but fail
#deimg1_pad = np.zeros((denoise_img1.shape[0]-2,denoise_img1.shape[1]-2,denoise_img1.shape[2]))
#for channel in range(3):
#    denoise_img1[0,:,channel], denoise_img1[-1,:,channel] = denoise_img1[-1,:,channel], denoise_img1[0,:,channel]
#    denoise_img1[:,0,channel], denoise_img1[:,-1,channel] = denoise_img1[:,-1,channel], denoise_img1[:,0,channel]
cv2.imwrite('output1.bmp',denoise_img1)

out2ori = PSNR(denoise_img1,img1_ori)
in2ori = PSNR(img1,img1_ori)

print in2ori
print out2ori


#processing img2

#img2 FFT
img2_freq = color_img_fft(img2)
img2_ori_freq = color_img_fft(img2_ori)

P2_pad = gauss_pad(P2,img2_freq.shape)
P2_freq = np.fft.fft2(P2_pad)

#kernel zero pad and FFT
motion_filter = mod_motion_process(33,130)
#motion_filter = motion_process(15,120)
motion_filter_pad = gauss_pad(motion_filter,img2_freq.shape)
motion_freq = np.fft.fft2(motion_filter_pad)

#deconvolve and ifft
#denoise_img2_freq = wiener_w_contrain(14,motion_freq,img2_freq,P2_freq)
denoise_img2_freq = weiner_filter(0.08,motion_freq,img2_freq)
denoise_img2 = color_img_ifft(denoise_img2_freq)
denoise_img2 = np.abs(denoise_img2)
denoise_test = denoise_img2
denoise_img2 = np.where(denoise_img2>255,255,denoise_img2)
denoise_img2 = np.where(denoise_img2<0,0,denoise_img2)
denoise_img2 = denoise_img2.astype(np.uint8)
cv2.imwrite('output2.bmp',denoise_img2)

out2ori = PSNR(denoise_img2,img2_ori)
in2ori = PSNR(img2,img2_ori)

print in2ori
print out2ori

#processing img3

#img3 FFT
img3_freq = color_img_fft(img3)
img3_ori_freq = color_img_fft(img3_ori)

P3_pad = gauss_pad(P3,img3_freq.shape)
P3_freq = np.fft.fft2(P3_pad)

#kernel zero pad and FFT
gauss_filter_1d = cv2.getGaussianKernel(ksize=35,sigma=4)
gauss_filter = np.outer(gauss_filter_1d,gauss_filter_1d)
gauss_filter_pad = gauss_pad(gauss_filter,img3_freq.shape)
gauss_freq = np.fft.fft2(gauss_filter_pad)

#deconvolve and ifft
denoise_img3_freq = wiener_w_contrain(5,gauss_freq,img3_freq,P3_freq)
#denoise_img3_freq = weiner_filter(0.02,gauss_freq,img3_freq)
denoise_img3 = color_img_ifft(denoise_img3_freq)
denoise_img3 = np.abs(denoise_img3)
denoise_test = denoise_img3
denoise_img3 = np.where(denoise_img3>255,255,denoise_img3)
denoise_img3 = np.where(denoise_img3<0,0,denoise_img3)
denoise_img3 = denoise_img3.astype(np.uint8)

cv2.imwrite('output3.bmp',denoise_img3)

#ori2ori = PSNR(img3_ori,img3_ori)
out2ori = PSNR(denoise_img3,img3_ori)
in2ori = PSNR(img3,img3_ori)

print in2ori
print out2ori

#test for kenel size
"""
test_img_freq = img2_ori_freq
test_img_freq[:,:,0] = img2_ori_freq[:,:,0]*motion_freq
test_img_freq[:,:,1] = img2_ori_freq[:,:,1]*motion_freq
test_img_freq[:,:,2] = img2_ori_freq[:,:,2]*motion_freq

test_img = color_img_ifft(test_img_freq)
test_img = np.abs(test_img).astype(np.uint8)
test_img = np.where(test_img>255,255,test_img)
test_img = np.where(test_img<0,0,test_img)
cv2.imwrite('test.bmp',test_img)
"""

"""
img1_freq = color_img_fft(img1)
test_img = color_img_ifft(img1_freq)
test_img = np.abs(test_img).astype(np.uint8)
cv2.imwrite('output1.bmp',test_img)


#test_img = cv2.GaussianBlur(img1_ori,(61,61),0)
#cv2.imwrite('output1.bmp',test_img)
"""