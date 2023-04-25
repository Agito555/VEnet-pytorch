import os
import numpy as np

ROOT_DIR='/home/dev/lzj/votenet'
SPLIT_SET='train'
DATA_PATH=os.path.join(ROOT_DIR,'sunrgbd/sunrgbd_pc_bbox_votes_50k_v1_%s'%(SPLIT_SET))

def pc_in_bx(point_cloud,bboxes,point_votes):
    npoints=point_cloud.shape[0]
    nbboxes=bboxes.shape[0]

    instance_label=np.zeros((npoints,1))

    foreground_flag=point_votes[:,0]
    foreground_flag=foreground_flag>0

    x,y,z=point_cloud[:,0],point_cloud[:,1],point_cloud[:,2]
    for i in range(nbboxes):
        cx,cy,cz=bboxes[i,0:3]
        l,w,h=bboxes[i,3:6] # 1/2
        angle=bboxes[i,6] # pi    

        cosa=np.cos(angle)
        sina=np.sin(angle)

        x_rot = np.round((x - cx) * cosa + (y - cy) * (-sina),decimals=6)
        y_rot = np.round((x - cx) * sina + (y - cy) * cosa,decimals=6)
        dz=np.round(z-cz,decimals=6)

        # x_rot = (x - cx) * cosa + (y - cy) * (-sina)
        # y_rot = (x - cx) * sina + (y - cy) * cosa
        # dz=z-cz

        inds=(x_rot >= -l ) * (x_rot <= l ) * (y_rot >= -w ) * (y_rot <= w) * ( dz >= -h) * ( dz <= h) * foreground_flag

        # for debug
        # print(l)
        # print(w)
        # print(h)
        # print(x_rot[41294])
        # print(y_rot[41294])
        # print(dz[41294])

        instance_label[inds]=i+1

    return instance_label


scan_names = sorted(list(set([os.path.basename(x)[0:6] \
            for x in os.listdir(DATA_PATH)])))

for scan_name in scan_names:

    # scan_name='007454'

    point_cloud = np.load(os.path.join(DATA_PATH, scan_name)+'_pc.npz')['pc'] # Nx6
    bboxes = np.load(os.path.join(DATA_PATH, scan_name)+'_bbox.npy')
    point_votes = np.load(os.path.join(DATA_PATH, scan_name)+'_votes.npz')['point_votes']

    instance_labels=pc_in_bx(point_cloud,bboxes,point_votes)

    kk=np.where(instance_labels)
    foreground=np.where(point_votes[:,0])

    print('scan_name is %s'%(scan_name))

    assert len(foreground[0])==len(kk[0]),'there is a bug'
    assert (foreground[0]==kk[0]).all(),'there is a bug'

    # np.save(os.path.join(DATA_PATH, scan_name)+'_instance_label.npy',instance_labels)

    k=1

print('all scenes are over')
