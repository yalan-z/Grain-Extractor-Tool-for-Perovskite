from torch import nn
import numpy as np
import cv2
from skimage import data, measure, morphology
import plotly
import plotly.express as px
import plotly.graph_objects as go

root='test_image'
figroot='test_image'

pathiou=open(root+'\\'+'IOU.txt','w',encoding="utf-8")
truthname='truth.png'
imgraw='result.png'
imgopen='lightopen.png'
imgseg='final.png'
preimg=cv2.imread(root+'\\'+imgraw,0)
truthimg=cv2.imread(root+'\\'+truthname,0)
num_labels, truthlabel, stats, centroids = cv2.connectedComponentsWithStats(truthimg, connectivity=4)
openimg=cv2.imread(root+'\\'+imgopen,0)
segimg=cv2.imread(root+'\\'+imgseg,0)

def webvisual(img,label,props,properties):
    fig = px.imshow(img, binary_string=True)
    fig.update_traces(hoverinfo='skip')  # hover is only for label info
    for index in range(1, label.max()):
        label_i = props[index].label

        contour = measure.find_contours(label == label_i, 0.5)[0]
        y, x = contour.T
        hoverinfo = ''

        hoverinfo += f'<b>{properties}: {getattr(props[index], properties)}</b><br>'
        fig.add_trace(go.Scatter(
            x=x, y=y, name=label_i,
            mode='lines', fill='toself', showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))

    plotly.io.show(fig)
    return
def IOU(f1,f2):
    numor=np.count_nonzero(np.bitwise_or(f1,f2))
    numand=np.count_nonzero(np.bitwise_and(f1,f2))
    error=numand/numor
    return error

propraw = measure.regionprops(preimg)
proptruth=measure.regionprops(truthlabel)
webvisual(cv2.imread(figroot+'\\'+'segresultcontrol1.png'),preimg,propraw,'area')
webvisual(truthimg,truthlabel,proptruth,'area')
ret, bipre = cv2.threshold(preimg, 0, 1, cv2.THRESH_BINARY)
ret, biopen = cv2.threshold(openimg, 0, 1, cv2.THRESH_BINARY)
ret, bitru = cv2.threshold(truthimg, 0, 1, cv2.THRESH_BINARY)
print('ML',IOU(bitru,bipre),file=pathiou)
print('open',IOU(bitru,biopen),file=pathiou)
pathiou.close()
