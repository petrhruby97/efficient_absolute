# select the data
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import h5py
import cv2
#import pygcransac
#import src.pybind_ransac
import poselib
import time
import statistics

from copy import deepcopy

DIR = "./oxford_data/"
seqs = ['model_house', 'corridor', 'merton1', 'merton2', 'merton3', 'library', 'wadham']
nums_imgs = [10, 11, 4, 4, 4, 4, 6]
shorts = ['house.', 'bt.', '', '', '', '', '']
#seq = 'model_house'
#short = 'house'

#seqs = ['wadham']
#nums_imgs = [6]
#shorts = ['']

def quaternion_from_matrix(matrix, isprecise=False):
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q

def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        print(R_gt, t_gt, R, t, q_gt)
        import IPython
        IPython.embed()

    return err_q, err_t

def check_points(x,X,R,t,K,thr):
    ret = []
    for i in range(x.shape[0]):
        proj = K@((R@X[i,:])+t)
        proj2 = proj[0:2]/proj[2]
        print(proj[2])
        err = (np.linalg.norm(proj2-x[i,:]))
        print(err)
        print()
        if err < thr:
            ret.append(True)
        else:
            ret.append(False)
    return ret

def check_lines(x1,x2,X1,X2,R,t,K,thr):
    ret = []
    for i in range(x1.shape[0]):
        proj1 = K@((R@X1[i,:])+t)
        proj2 = K@((R@X2[i,:])+t)
        l = np.cross(proj1, proj2)
        l = l/np.linalg.norm(l[0:2])
        err_1 = np.abs(l[0]*x1[i,0] + l[1]*x1[i,1] + l[2])
        err_2 = np.abs(l[0]*x2[i,0] + l[1]*x2[i,1] + l[2])
        if err_1 < thr and err_1 < thr:
            ret.append(True)
        else:
            ret.append(False)
    return ret

# Setup statistics
count = 0
below_5deg = 0
below_10deg = 0
below_20deg = 0
rotations = []
translations = []

# PARAMS
SNN_threshold = 0.85

for i in range(7):
    seq = seqs[i]
    num_imgs = nums_imgs[i]
    short = shorts[i]

    print("***********")
    print(seq)
    print("***********")

    if seq == 'corridor' or seq == 'model_house':
        anfang = int(0)
    else:
        anfang = int(1)

    #statistics:
    times_2p1l_our = []
    rot_2p1l_our = []
    tran_2p1l_our = []

    times_2p1l_3Q3 = []
    rot_2p1l_3Q3 = []
    tran_2p1l_3Q3 = []

    times_2p1l_bouaziz = []
    rot_2p1l_bouaziz = []
    tran_2p1l_bouaziz = []

    times_2p1l_bouaziz_lu = []
    rot_2p1l_bouaziz_lu = []
    tran_2p1l_bouaziz_lu = []

    times_1p2l_our = []
    rot_1p2l_our = []
    tran_1p2l_our = []

    times_1p2l_3Q3 = []
    rot_1p2l_3Q3 = []
    tran_1p2l_3Q3 = []

    #print(seq)
    for img_id in range(anfang, num_imgs):
        print("Image " + str(img_id))

        #load matches
        #0 is present, so they are indexing from zero
        pt_matches = np.loadtxt(DIR + seq + "/2D/" + short + "nview-corners")
        #the ID of the 3D point is given by the position, the ID of the 2D point is given by the number
        line_matches = np.loadtxt(DIR + seq + "/2D/" + short + "nview-lines")

        #load 3D points and lines
        p3d = np.loadtxt(DIR + seq + "/3D/" + short + "p3d")
        l3d = np.loadtxt(DIR + seq + "/3D/" + short + "l3d")

        #load the GT pose
        if seq == 'corridor' or seq == 'model_house':
            P = np.loadtxt(DIR + seq + "/3D/" + short + "" + str(img_id).zfill(3) + ".P")
        else:
            P = np.loadtxt(DIR + seq + "/2D/" + short + "" + str(img_id).zfill(3) + ".P")

        #load the 2D points and lines
        p2d = np.loadtxt(DIR + seq + "/2D/" + short + "" + str(img_id).zfill(3) + ".corners")
        l2d = np.loadtxt(DIR + seq + "/2D/" + short + "" + str(img_id).zfill(3) + ".lines")

        #extract the matches for the particular image
        pt_3D_ix = np.array(list(range(pt_matches.shape[0])))[pt_matches[:,img_id-anfang]>=0]
        pt_2D_ix = pt_matches[ pt_matches[:,img_id-anfang]>=0 ,img_id-anfang].astype(int)
        ln_3D_ix = np.array(list(range(line_matches.shape[0])))[line_matches[:,img_id-anfang]>=0]
        ln_2D_ix = line_matches[ line_matches[:,img_id-anfang]>=0 ,img_id-anfang].astype(int)

        #create the matches
        pt_2D = p2d[pt_2D_ix,:]
        pt_3D = p3d[pt_3D_ix,:]
        ln_3D = l3d[ln_3D_ix,:]
        ln_2D = l2d[ln_2D_ix,:]

        #extract the line endpoints
        X1_3D = ln_3D[:,0:3]
        X2_3D = ln_3D[:,3:6]
        x1_2D = ln_2D[:,0:2]
        x2_2D = ln_2D[:,2:4]

        #decompose the camera matrix
        K0,R = linalg.rq(P[:,0:3])
        K = K0/K0[2,2]
        P = P/K0[2,2]

        #print(K)

        cor = np.eye(3)
        if K[0,0] < 0:
            cor[0,0] = -1
        if K[1,1] < 0:
            cor[1,1] = -1
        K = K@cor
        R = cor@R

        Ki = linalg.inv(K)
        t = Ki @ P[:,3]
        Pc = Ki @ P

        if np.linalg.det(R) < 0:
            R = -1*R
            if seq=='wadham':
                t = -1*t
            else:
                pt_3D = -1*pt_3D
                X1_3D = -1*X1_3D
                X2_3D = -1*X2_3D
            #TODO plot the images and the 2D/3D points to see what is the correct way to address the negative depths

        #inl_mask = check_points(pt_2D,pt_3D,R,t,K,3.0)
        #print(inl_mask)
        #print(np.sum(np.array(inl_mask)))
        #exit()

        K_simp = np.eye(3)
        K_simp[0,0] = K[0,0]
        K_simp[1,1] = K[1,1]
        K_simp[0,2] = K[0,2]
        K_simp[1,2] = K[1,2]

        #solve the problem
        camera_params = [K[0,0], K[1,1], K[0,2], K[1,2]]
        camera = {'model': 'PINHOLE', 'width': int(2*K[0,2]), 'height': int(2*K[1,2]), 'params': camera_params}

        print("OUR 2P1L")
        ransac_opt = {"seed": 19021997}
        start = time.time()
        res = poselib.estimate_absolute_pose_pnpl(pt_2D, pt_3D, x1_2D, x2_2D, X1_3D, X2_3D, camera, ransac_opt)
        end = time.time()
        print(end - start)
        #print(res[1]["refinements"])
        R_est = res[0].R
        t_est = res[0].t
        rot_err, tr_err = evaluate_R_t(R, t, R_est, t_est)
        #print(str(rot_err) + " " + str(tr_err))
        #print()
        times_2p1l_our.append(end-start)
        rot_2p1l_our.append(rot_err)
        tran_2p1l_our.append(tr_err)


        print("3Q3 2P1L")
        ransac_opt = {"seed": 19021997}
        start = time.time()
        res = poselib.estimate_absolute_pose_pnpl_3Q3(pt_2D, pt_3D, x1_2D, x2_2D, X1_3D, X2_3D, camera, ransac_opt)
        end = time.time()
        print(end - start)
        #print(res[1]["refinements"])
        #print(res)
        R_est = res[0].R
        t_est = res[0].t
        rot_err, tr_err = evaluate_R_t(R, t, R_est, t_est)
        #print(str(rot_err) + " " + str(tr_err))
        #print()
        times_2p1l_3Q3.append(end-start)
        rot_2p1l_3Q3.append(rot_err)
        tran_2p1l_3Q3.append(tr_err)


        print("Bouaziz 2P1L")
        ransac_opt = {"seed": 19021997}
        start = time.time()
        res = poselib.estimate_absolute_pose_pnpl_bouaziz(pt_2D, pt_3D, x1_2D, x2_2D, X1_3D, X2_3D, camera, ransac_opt)
        end = time.time()
        print(end - start)
        #print(res[1]["refinements"])
        #print(res)
        R_est = res[0].R
        t_est = res[0].t
        rot_err, tr_err = evaluate_R_t(R, t, R_est, t_est)
        #print(str(rot_err) + " " + str(tr_err))
        #print()
        times_2p1l_bouaziz.append(end-start)
        rot_2p1l_bouaziz.append(rot_err)
        tran_2p1l_bouaziz.append(tr_err)

        print("Bouaziz LU 2P1L")
        ransac_opt = {"seed": 19021997}
        start = time.time()
        res = poselib.estimate_absolute_pose_pnpl_bouaziz_lu(pt_2D, pt_3D, x1_2D, x2_2D, X1_3D, X2_3D, camera, ransac_opt)
        end = time.time()
        print(end - start)
        #print(res[1]["refinements"])
        #print(res)
        R_est = res[0].R
        t_est = res[0].t
        rot_err, tr_err = evaluate_R_t(R, t, R_est, t_est)
        #print(str(rot_err) + " " + str(tr_err))
        #print()
        times_2p1l_bouaziz_lu.append(end-start)
        rot_2p1l_bouaziz_lu.append(rot_err)
        tran_2p1l_bouaziz_lu.append(tr_err)

        print("OUR 1P2L")
        ransac_opt = {"seed": 19021997}
        start = time.time()
        res = poselib.estimate_absolute_pose_pnpl_1P2L(pt_2D, pt_3D, x1_2D, x2_2D, X1_3D, X2_3D, camera, ransac_opt)
        end = time.time()
        print(end - start)
        #print(res[1]["refinements"])
        #print(res)
        R_est = res[0].R
        t_est = res[0].t
        rot_err, tr_err = evaluate_R_t(R, t, R_est, t_est)
        #print(str(rot_err) + " " + str(tr_err))
        #print()
        times_1p2l_our.append(end-start)
        rot_1p2l_our.append(rot_err)
        tran_1p2l_our.append(tr_err)


        print("3Q3 1P2L")
        ransac_opt = {"seed": 19021997}
        start = time.time()
        res = poselib.estimate_absolute_pose_pnpl_1P2L_3Q3(pt_2D, pt_3D, x1_2D, x2_2D, X1_3D, X2_3D, camera, ransac_opt)
        end = time.time()
        print(end - start)
        #print(res[1]["refinements"])
        R_est = res[0].R
        t_est = res[0].t
        rot_err, tr_err = evaluate_R_t(R, t, R_est, t_est)
        #print(str(rot_err) + " " + str(tr_err))
        #print()
        times_1p2l_3Q3.append(end-start)
        rot_1p2l_3Q3.append(rot_err)
        tran_1p2l_3Q3.append(tr_err)

        print()

        #exit()

    print()
    print()
    print("* * * * * * * * * * * * * * * ")
    print("* R * E * S * U * L * T * S * ")
    print("* * * * * * * * * * * * * * * ")

    print("* T * I * M * E *")
    print("OUR 2P1L")
    print(times_2p1l_our)
    print(statistics.mean(times_2p1l_our))
    print(statistics.mean(times_2p1l_our) / statistics.mean(times_2p1l_3Q3))
    print()

    print("3Q3 2P1L")
    print(times_2p1l_3Q3)
    print(statistics.mean(times_2p1l_3Q3))
    print(statistics.mean(times_2p1l_3Q3) / statistics.mean(times_2p1l_3Q3))
    print()

    print("BOUAZIZ 2P1L")
    print(times_2p1l_bouaziz)
    print(statistics.mean(times_2p1l_bouaziz))
    print(statistics.mean(times_2p1l_bouaziz) / statistics.mean(times_2p1l_3Q3))
    print()

    print("BOUAZIZ LU 2P1L")
    print(times_2p1l_bouaziz_lu)
    print(statistics.mean(times_2p1l_bouaziz_lu))
    print(statistics.mean(times_2p1l_bouaziz_lu) / statistics.mean(times_2p1l_3Q3))
    print()

    print("OUR 2P1L")
    print(times_1p2l_our)
    print(statistics.mean(times_1p2l_our))
    print(statistics.mean(times_1p2l_our) / statistics.mean(times_1p2l_3Q3))
    print()

    print("3Q3 2P1L")
    print(times_1p2l_3Q3)
    print(statistics.mean(times_1p2l_3Q3))
    print(statistics.mean(times_1p2l_3Q3) / statistics.mean(times_1p2l_3Q3))
    print()


    print("* R * O * T *")
    print("OUR 2P1L")
    print(rot_2p1l_our)
    print(statistics.mean(rot_2p1l_our))
    print()

    print("3Q3 2P1L")
    print(rot_2p1l_3Q3)
    print(statistics.mean(rot_2p1l_3Q3))
    print()

    print("BOUAZIZ 2P1L")
    print(rot_2p1l_bouaziz)
    print(statistics.mean(rot_2p1l_bouaziz))
    print()

    print("BOUAZIZ LU 2P1L")
    print(rot_2p1l_bouaziz_lu)
    print(statistics.mean(rot_2p1l_bouaziz_lu))
    print()

    print("OUR 2P1L")
    print(rot_1p2l_our)
    print(statistics.mean(rot_1p2l_our))
    print()

    print("3Q3 2P1L")
    print(rot_1p2l_3Q3)
    print(statistics.mean(rot_1p2l_3Q3))
    print()

    print()


    print("* T * R * A * N *")
    print("OUR 2P1L")
    print(tran_2p1l_our)
    print(statistics.mean(tran_2p1l_our))
    print()

    print("3Q3 2P1L")
    print(tran_2p1l_3Q3)
    print(statistics.mean(tran_2p1l_3Q3))
    print()

    print("BOUAZIZ 2P1L")
    print(tran_2p1l_bouaziz)
    print(statistics.mean(tran_2p1l_bouaziz))
    print()

    print("BOUAZIZ LU 2P1L")
    print(tran_2p1l_bouaziz_lu)
    print(statistics.mean(tran_2p1l_bouaziz_lu))
    print()

    print("OUR 2P1L")
    print(tran_1p2l_our)
    print(statistics.mean(tran_1p2l_our))
    print()

    print("3Q3 2P1L")
    print(tran_1p2l_3Q3)
    print(statistics.mean(tran_1p2l_3Q3))
    print()

print()




