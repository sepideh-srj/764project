# from __future__ import print_function, division
import numpy as np
import math
import torch
import os
from matplotlib import pyplot as plt
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from torch.utils import data
from scipy.io import loadmat, savemat
from enum import Enum
from grassdata import GRASSDataset, Tree, unrotate_boxes, rotate_boxes_only
from dataset import SCORESTest
import draw3dOBB
import testVQContext
import testVQContextGen
import torch.utils.data
import math

## Some function definitions for box decoding

def renderBoxes2mesh(boxes, gtboxs, obj_names):
    results = []
    for box_i in range(len(boxes)):
        vertices = []
        faces = []
        obj_name = obj_names[box_i]
        v_num = 0
        for name in obj_name:
            with open(os.path.join('../../partNet_objs/', name), 'r') as f:
                lines = f.readlines()
            t = 0
            for line in lines:
                if line[0] != 'v' and line[0] != 'f':
                    continue
                line = line.strip('\n')
                items = line.split(' ')
                if items[0] == 'v':
                    vertices.append((float(items[1]), float(items[2]), float(items[3])))
                    t += 1
                if items[0] == 'f':
                    faces.append([int(items[1])+v_num, int(items[2])+v_num, int(items[3])+v_num])
            v_num += t
        if isinstance(gtboxs[box_i], int):
            results.append((vertices, faces))
        else:
            gtbox = gtboxs[box_i].cpu().numpy().squeeze()
            gtCenter = gtbox[0:3][np.newaxis, ...].T
            gtlengths = gtbox[3:6]
            gtdir_1 = gtbox[6:9]
            gtdir_2 = gtbox[9:12]
            gtdir_1 = gtdir_1/LA.norm(gtdir_1)
            gtdir_2 = gtdir_2/LA.norm(gtdir_2)
            gtdir_3 = np.cross(gtdir_1, gtdir_2)
            # gtdir_3 = gtdir_3/LA.norm(gtdir_3)


            predbox = boxes[box_i].cpu().numpy().squeeze()
            predCenter = predbox[0:3][np.newaxis, ...].T
            predlengths = predbox[3:6]
            preddir_1 = predbox[6:9]
            preddir_2 = predbox[9:12]
            preddir_1 = preddir_1/LA.norm(preddir_1)
            preddir_2 = preddir_2/LA.norm(preddir_2)
            preddir_3 = -np.cross(preddir_1, preddir_2)
            # preddir_3 = preddir_3/LA.norm(preddir_3)


            scale = predlengths / gtlengths
            scale = np.array([[scale[2], 0, 0], [0, scale[0], 0], [0, 0, scale[1]]])
            x = np.array(vertices).T
            A = np.array([gtdir_1, gtdir_2, gtdir_3])
            B = np.array([preddir_1, preddir_2, preddir_3])
            B = B.T
            y = scale.dot(B).dot(A).dot(x-gtCenter)+predCenter
            x = y.T
            vertices = []
            for i in range(x.shape[0]):
                vertices.append(x[i])
            for i in range(len(faces)):
                temp = faces[i][0]
                faces[i][0] = faces[i][1]
                faces[i][1] = temp
            results.append((vertices, faces))

    return results

def renderBoxes2mesh_new(boxes, gtboxs, obj_names):
    results = []
    for box_i in range(len(boxes)):
        vertices = []
        faces = []
        obj_name = obj_names[box_i]
        v_num = 0
        for name in obj_name:
            with open(os.path.join('../../partNet_objs/', name), 'r') as f:
                lines = f.readlines()
            t = 0
            for line in lines:
                if line[0] != 'v' and line[0] != 'f':
                    continue
                line = line.strip('\n')
                items = line.split(' ')
                if items[0] == 'v':
                    vertices.append((float(items[1]), float(items[2]), float(items[3])))
                    t += 1
                if items[0] == 'f':
                    faces.append([int(items[1])+v_num, int(items[2])+v_num, int(items[3])+v_num])
            v_num += t

        gtbox = gtboxs[box_i].cpu().numpy().squeeze()
        gtCenter = gtbox[0:3][np.newaxis, ...].T
        gtlengths = gtbox[3:6]
        gtdir_1 = gtbox[6:9]
        gtdir_2 = gtbox[9:12]
        gtdir_1 = gtdir_1/LA.norm(gtdir_1)
        gtdir_2 = gtdir_2/LA.norm(gtdir_2)
        gtdir_3 = np.cross(gtdir_1, gtdir_2)
        # gtdir_3 = gtdir_3/LA.norm(gtdir_3)


        predbox = boxes[box_i].cpu().numpy().squeeze()
        predCenter = predbox[0:3][np.newaxis, ...].T
        predlengths = predbox[3:6]
        preddir_1 = predbox[6:9]
        preddir_2 = predbox[9:12]
        preddir_1 = preddir_1/LA.norm(preddir_1)
        preddir_2 = preddir_2/LA.norm(preddir_2)
        preddir_3 = -np.cross(preddir_1, preddir_2)
        # preddir_3 = preddir_3/LA.norm(preddir_3)


        scale = predlengths / gtlengths
        scale = np.array([[scale[2], 0, 0], [0, scale[0], 0], [0, 0, scale[1]]])
        x = np.array(vertices).T
        A = np.array([gtdir_1, gtdir_2, gtdir_3])
        B = np.array([preddir_1, preddir_2, preddir_3])
        B = B.T
        # y = x - gtCenter
        # y = np.dot(scale,y)
        # y = np.dot(A,y)
        # y = np.dot(B,y)
        # y = y + predCenter
        y = scale.dot(B).dot(A).dot(x-gtCenter)+predCenter
        x = y.T
        vertices = []
        for i in range(x.shape[0]):
            vertices.append(x[i])
        for i in range(len(faces)):
            temp = faces[i][0]
            faces[i][0] = faces[i][1]
            faces[i][1] = temp
        results.append((vertices, faces))

    return results


def saveOBJ(obj_names, outfilename, results):
    cmap = plt.get_cmap('jet_r')
    f = open(outfilename, 'w')
    offset = 0
    for box_i in range(len(results)):
        color = [0.5, 0.5, 1, 1]#cmap(box_i / len(results))[:-1]
        vertices = results[box_i][0]
        faces = results[box_i][1]
        for i in range(len(vertices)):
            f.write('v ' + str(vertices[i][0]) + ' ' + str(vertices[i][1]) + ' ' + str(vertices[i][2]) +
                    ' ' + str(color[0]) + ' ' + str(color[1]) + ' ' + str(color[2]) + ' ' + str(color[3]) + '\n')
        for i in range(len(faces)):
            f.write('f ' + str(faces[i][0]+offset) + ' ' + str(faces[i][1]+offset) + ' ' + str(faces[i][2]+offset) + '\n')
        offset += len(vertices)
    f.close()

def alignBoxAndRender(gtBoxes, predBoxes, boxes_type, obj_names, outfilename):
    results = renderBoxes2mesh(gtBoxes, boxes_type, obj_names)
    for i in range(len(results)):
        gtbox = gtBoxes[i].cpu().numpy().squeeze()
        gtCenter = gtbox[0:3][np.newaxis, ...].T
        gtlengths = gtbox[3:6]
        gtdir_1 = gtbox[6:9]
        gtdir_2 = gtbox[9:12]
        gtdir_1 = gtdir_1/LA.norm(gtdir_1)
        gtdir_2 = gtdir_2/LA.norm(gtdir_2)
        gtdir_3 = np.cross(gtdir_1, gtdir_2)

        predbox = predBoxes[i].cpu().numpy().squeeze()
        predCenter = predbox[0:3][np.newaxis, ...].T
        predlengths = predbox[3:6]
        preddir_1 = predbox[6:9]
        preddir_2 = predbox[9:12]
        preddir_1 = preddir_1/LA.norm(preddir_1)
        preddir_2 = preddir_2/LA.norm(preddir_2)
        preddir_3 = np.cross(preddir_1, preddir_2)

        scale = predlengths / gtlengths
        scale = np.array([[scale[0], 0, 0], [0, scale[1], 0], [0, 0, scale[2]]])
        x = np.array(results[i][0]).T
        A = np.array([gtdir_1, gtdir_2, gtdir_3])
        B = np.array([preddir_1, preddir_2, preddir_3])
        B = B.T
        y = scale.dot(B).dot(A).dot(x-gtCenter)+predCenter
        x = y.T
        for t in range(len(results[i][0])):
            results[i][0][t] = x[t]
    saveOBJ(obj_names, outfilename, results)



def tryPlot():
    cmap = plt.get_cmap('jet_r')
    fig = plt.figure()
    ax = Axes3D(fig)
    draw(ax, [-0.0152730000000000, -0.113074400000000, 0.00867852000000000, 0.766616000000000, 0.483920000000000,
              0.0964542000000000,
              8.65505000000000e-06, -0.000113369000000000, 0.999997000000000, 0.989706000000000, 0.143116000000000,
              7.65900000000000e-06], cmap(float(1) / 7))
    draw(ax, [-0.310188000000000, 0.188456800000000, 0.00978854000000000, 0.596362000000000, 0.577190000000000,
              0.141414800000000,
              -0.331254000000000, 0.943525000000000, 0.00456327000000000, -0.00484978000000000, -0.00653891000000000,
              0.999967000000000], cmap(float(2) / 7))
    draw(ax, [-0.290236000000000, -0.334664000000000, -0.328648000000000, 0.322898000000000, 0.0585966000000000,
              0.0347996000000000,
              -0.330345000000000, -0.942455000000000, 0.0514932000000000, 0.0432524000000000, 0.0393726000000000,
              0.998095000000000], cmap(float(3) / 7))
    draw(ax, [-0.289462000000000, -0.334842000000000, 0.361558000000000, 0.322992000000000, 0.0593536000000000,
              0.0350418000000000,
              0.309240000000000, 0.949730000000000, 0.0485183000000000, -0.0511885000000000, -0.0343219000000000,
              0.998099000000000], cmap(float(4) / 7))
    draw(ax, [0.281430000000000, -0.306584000000000, 0.382928000000000, 0.392156000000000, 0.0409424000000000,
              0.0348472000000000,
              0.322342000000000, -0.942987000000000, 0.0828920000000000, -0.0248683000000000, 0.0791002000000000,
              0.996556000000000], cmap(float(5) / 7))
    draw(ax, [0.281024000000000, -0.306678000000000, -0.366110000000000, 0.392456000000000, 0.0409366000000000,
              0.0348446000000000,
              -0.322608000000000, 0.942964000000000, 0.0821142000000000, 0.0256742000000000, -0.0780031000000000,
              0.996622000000000], cmap(float(6) / 7))
    draw(ax, [0.121108800000000, -0.0146729400000000, 0.00279166000000000, 0.681576000000000, 0.601756000000000,
              0.0959706000000000,
              -0.986967000000000, -0.160173000000000, 0.0155341000000000, 0.0146809000000000, 0.00650174000000000,
              0.999801000000000], cmap(float(7) / 7))
    plt.show()


def draw(ax, p, color):
    tmpPoint = p

    center = tmpPoint[0: 3]
    lengths = tmpPoint[3: 6]
    dir_1 = tmpPoint[6: 9]
    dir_2 = tmpPoint[9:]

    dir_1 = dir_1 / LA.norm(dir_1)
    dir_2 = dir_2 / LA.norm(dir_2)
    dir_3 = np.cross(dir_1, dir_2)
    dir_3 = dir_3 / LA.norm(dir_3)
    cornerpoints = np.zeros([8, 3])

    d1 = 0.5 * lengths[0] * dir_1
    d2 = 0.5 * lengths[1] * dir_2
    d3 = 0.5 * lengths[2] * dir_3

    cornerpoints[0][:] = center - d1 - d2 - d3
    cornerpoints[1][:] = center - d1 + d2 - d3
    cornerpoints[2][:] = center + d1 - d2 - d3
    cornerpoints[3][:] = center + d1 + d2 - d3
    cornerpoints[4][:] = center - d1 - d2 + d3
    cornerpoints[5][:] = center - d1 + d2 + d3
    cornerpoints[6][:] = center + d1 - d2 + d3
    cornerpoints[7][:] = center + d1 + d2 + d3

    ax.plot([cornerpoints[0][0], cornerpoints[1][0]], [cornerpoints[0][1], cornerpoints[1][1]],
            [cornerpoints[0][2], cornerpoints[1][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[2][0]], [cornerpoints[0][1], cornerpoints[2][1]],
            [cornerpoints[0][2], cornerpoints[2][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[3][0]], [cornerpoints[1][1], cornerpoints[3][1]],
            [cornerpoints[1][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[3][0]], [cornerpoints[2][1], cornerpoints[3][1]],
            [cornerpoints[2][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[5][0]], [cornerpoints[4][1], cornerpoints[5][1]],
            [cornerpoints[4][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[6][0]], [cornerpoints[4][1], cornerpoints[6][1]],
            [cornerpoints[4][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[5][0], cornerpoints[7][0]], [cornerpoints[5][1], cornerpoints[7][1]],
            [cornerpoints[5][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[6][0], cornerpoints[7][0]], [cornerpoints[6][1], cornerpoints[7][1]],
            [cornerpoints[6][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[4][0]], [cornerpoints[0][1], cornerpoints[4][1]],
            [cornerpoints[0][2], cornerpoints[4][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[5][0]], [cornerpoints[1][1], cornerpoints[5][1]],
            [cornerpoints[1][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[6][0]], [cornerpoints[2][1], cornerpoints[6][1]],
            [cornerpoints[2][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[3][0], cornerpoints[7][0]], [cornerpoints[3][1], cornerpoints[7][1]],
            [cornerpoints[3][2], cornerpoints[7][2]], c=color)


def showGenshapes(genshapes):
    for i in range(len(genshapes)):
        recover_boxes = genshapes[i]

        fig = plt.figure(i)
        cmap = plt.get_cmap('jet_r')
        ax = Axes3D(fig)
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.7, 0.7)
        ax.set_zlim(-0.7, 0.7)

        for jj in range(len(recover_boxes)):
            p = recover_boxes[jj][:]
            draw(ax, p, cmap(float(jj) / len(recover_boxes)))

        plt.show()


def showGenshape(genshape):
    recover_boxes = genshape

    fig = plt.figure(0)
    cmap = plt.get_cmap('jet_r')
    ax = Axes3D(fig)
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)
    fig.add_axes(ax)

    for jj in range(len(recover_boxes)):
        p = recover_boxes[jj][:].squeeze(0).numpy()
        draw(ax, p, cmap(float(jj) / len(recover_boxes)))

    plt.show()



# %%

def decode_boxes(root):
    """
    Decode a root code into a tree structure of boxes
    """
    syms = [torch.ones(8).mul(10)]
    stack = [root]
    boxes = []
    labels = []
    syms_out = []
    objs = []
    while len(stack) > 0:
        node = stack.pop()
        node_type = torch.LongTensor([node.node_type.value]).item()
        if node_type == 1:  # ADJ
            # left, right = model.adjDecoder(f)
            stack.append(node.left)
            stack.append(node.right)
            s = syms.pop()
            syms.append(s)
            syms.append(s)
        if node_type == 2:  # SYM
            # left, s = model.symDecoder(f)
            # s = s.squeeze(0)
            stack.append(node.left)
            syms.pop()
            syms.append(node.sym.squeeze(0))
            # print(node.sym.squeeze(0))
        if node_type == 0:  # BOX
            reBox = node.box
            s = syms.pop()
            boxes.append(reBox)
            syms_out.append(s)
            # print(node.label.item())
            labels.append(node.label)
            objs.append(node.objname)
    return boxes, syms_out, labels, objs


# %% md
# %%

def vrrotvec2mat(rotvector):
    s = math.sin(rotvector[3])
    c = math.cos(rotvector[3])
    t = 1 - c
    x = rotvector[0]
    y = rotvector[1]
    z = rotvector[2]
    # m = torch.FloatTensor([[t*x*x+c, t*x*y-s*z, t*x*z+s*y], [t*x*y+s*z, t*y*y+c, t*y*z-s*x], [t*x*z-s*y, t*y*z+s*x, t*z*z+c]]).cuda()
    m = torch.FloatTensor(
        [[t * x * x + c, t * x * y - s * z, t * x * z + s * y], [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
         [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])
    return m


def decode_structure(root):
    """
    Decode a root code into a tree structure of boxes
    """
    # decode = model.sampleDecoder(root_code)
    syms = [torch.ones(8).mul(10)]
    stack = [root]
    boxes = []
    objs = []
    copyboxs = []
    while len(stack) > 0:
        node = stack.pop()
        # label_prob = model.nodeClassifier(f)
        # _, label = torch.max(label_prob, 1)
        # label = node.label.item()
        node_type = torch.LongTensor([node.node_type.value]).item()
        if node_type == 1:  # ADJ
            # left, right = model.adjDecoder(f)
            stack.append(node.left)
            stack.append(node.right)
            s = syms.pop()
            syms.append(s)
            syms.append(s)
        if node_type == 2:  # SYM
            # left, s = model.symDecoder(f)
            # s = s.squeeze(0)
            stack.append(node.left)
            syms.pop()
            syms.append(node.sym.squeeze(0))
        if node_type == 0:  # BOX
            reBox = node.box
            reBoxes = [reBox]
            recopyBoxes = [1]
            reObj = node.objname
            reObjs = [reObj]
            s = syms.pop()
            l1 = abs(s[0] + 1)
            l2 = abs(s[0])
            l3 = abs(s[0] - 1)
            print(l1,l2,l3)
            if l1 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.data.squeeze(0), 1, 0)
                f1 = torch.cat([sList[1], sList[2], sList[3]])
                f1 = f1 / torch.norm(f1)
                f2 = torch.cat([sList[4], sList[5], sList[6]])
                folds = round(1 / s[7].item())
                for i in range(folds - 1):
                    rotvector = torch.cat([f1, sList[7].mul(2 * 3.1415).mul(i + 1)])
                    rotm = vrrotvec2mat(rotvector)
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    dir0 = torch.cat([bList[3], bList[4], bList[5]])
                    dir1 = torch.cat([bList[6], bList[7], bList[8]])
                    dir2 = torch.cat([bList[9], bList[10], bList[11]])
                    newcenter = rotm.matmul(center.add(-f2)).add(f2)
                    newdir1 = rotm.matmul(dir1)
                    newdir2 = rotm.matmul(dir2)
                    newbox = torch.cat([newcenter, dir0, newdir1, newdir2])
                    reBoxes.append(newbox)
                    recopyBoxes.append(reBox)
                    reObjs.append(reObj)
            if l3 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.data.squeeze(0), 1, 0)
                trans = torch.cat([sList[1], sList[2], sList[3]])
                trans_end = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                trans_length = math.sqrt(torch.sum(trans ** 2))
                trans_total = math.sqrt(torch.sum(trans_end.add(-center) ** 2))
                folds = round(trans_total / trans_length)
                for i in range(folds):
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    dir0 = torch.cat([bList[3], bList[4], bList[5]])
                    dir1 = torch.cat([bList[6], bList[7], bList[8]])
                    dir2 = torch.cat([bList[9], bList[10], bList[11]])
                    newcenter = center.add(trans.mul(i + 1))
                    newbox = torch.cat([newcenter, dir0, dir1, dir2])
                    reBoxes.append(newbox)
                    recopyBoxes.append(reBox)
                    reObjs.append(reObj)
            if l2 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.data.squeeze(0), 1, 0)
                ref_normal = torch.cat([sList[1], sList[2], sList[3]])
                ref_normal = ref_normal / torch.norm(ref_normal)
                ref_point = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                dir0 = torch.cat([bList[3], bList[4], bList[5]])
                dir1 = torch.cat([bList[6], bList[7], bList[8]])
                dir2 = torch.cat([bList[9], bList[10], bList[11]])
                if ref_normal.matmul(ref_point.add(-center)) < 0:
                    ref_normal = -ref_normal
                newcenter = ref_normal.mul(2 * abs(torch.sum(ref_point.add(-center).mul(ref_normal)))).add(center)
                if ref_normal.matmul(dir1) < 0:
                    ref_normal = -ref_normal
                dir1 = dir1.add(ref_normal.mul(-2 * ref_normal.matmul(dir1)))
                if ref_normal.matmul(dir2) < 0:
                    ref_normal = -ref_normal
                dir2 = dir2.add(ref_normal.mul(-2 * ref_normal.matmul(dir2)))
                newbox = torch.cat([newcenter, dir0, dir1, dir2])
                reBoxes.append(newbox)
                recopyBoxes.append(reBox)
                reObjs.append(reObj)

            boxes.extend(reBoxes)
            objs.extend(reObjs)
            copyboxs.extend(recopyBoxes)
    return boxes, copyboxs, objs


# %%

def decode_structure_with_labels(root):
    """
    Decode a root code into a tree structure of boxes
    """
    # decode = model.sampleDecoder(root_code)
    syms = [torch.ones(8).mul(10)]
    stack = [root]
    boxes = []
    labels = []
    while len(stack) > 0:
        node = stack.pop()
        # label_prob = model.nodeClassifier(f)
        # _, label = torch.max(label_prob, 1)
        # label = node.label.item()
        node_type = torch.LongTensor([node.node_type.value]).item()
        if node_type == 1:  # ADJ
            # left, right = model.adjDecoder(f)
            stack.append(node.left)
            stack.append(node.right)
            s = syms.pop()
            syms.append(s)
            syms.append(s)
        if node_type == 2:  # SYM
            # left, s = model.symDecoder(f)
            # s = s.squeeze(0)
            stack.append(node.left)
            syms.pop()
            syms.append(node.sym.squeeze(0))
        if node_type == 0:  # BOX
            reBox = node.box
            reBoxes = [reBox]
            label = node.label.item()
            s = syms.pop()
            l1 = abs(s[0] + 1)
            l2 = abs(s[0])
            l3 = abs(s[0] - 1)
            if l1 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.data.squeeze(0), 1, 0)
                f1 = torch.cat([sList[1], sList[2], sList[3]])
                f1 = f1 / torch.norm(f1)
                f2 = torch.cat([sList[4], sList[5], sList[6]])
                folds = round(1 / s[7].item())
                for i in range(folds - 1):
                    rotvector = torch.cat([f1, sList[7].mul(2 * 3.1415).mul(i + 1)])
                    rotm = vrrotvec2mat(rotvector)
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    dir0 = torch.cat([bList[3], bList[4], bList[5]])
                    dir1 = torch.cat([bList[6], bList[7], bList[8]])
                    dir2 = torch.cat([bList[9], bList[10], bList[11]])
                    newcenter = rotm.matmul(center.add(-f2)).add(f2)
                    newdir1 = rotm.matmul(dir1)
                    newdir2 = rotm.matmul(dir2)
                    newbox = torch.cat([newcenter, dir0, newdir1, newdir2])
                    reBoxes.append(newbox)
            if l3 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.data.squeeze(0), 1, 0)
                trans = torch.cat([sList[1], sList[2], sList[3]])
                trans_end = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                trans_length = math.sqrt(torch.sum(trans ** 2))
                trans_total = math.sqrt(torch.sum(trans_end.add(-center) ** 2))
                folds = round(trans_total / trans_length)
                for i in range(folds):
                    center = torch.cat([bList[0], bList[1], bList[2]])
                    dir0 = torch.cat([bList[3], bList[4], bList[5]])
                    dir1 = torch.cat([bList[6], bList[7], bList[8]])
                    dir2 = torch.cat([bList[9], bList[10], bList[11]])
                    newcenter = center.add(trans.mul(i + 1))
                    newbox = torch.cat([newcenter, dir0, dir1, dir2])
                    reBoxes.append(newbox)
            if l2 < 0.15:
                sList = torch.split(s, 1, 0)
                bList = torch.split(reBox.data.squeeze(0), 1, 0)
                ref_normal = torch.cat([sList[1], sList[2], sList[3]])
                ref_normal = ref_normal / torch.norm(ref_normal)
                ref_point = torch.cat([sList[4], sList[5], sList[6]])
                center = torch.cat([bList[0], bList[1], bList[2]])
                dir0 = torch.cat([bList[3], bList[4], bList[5]])
                dir1 = torch.cat([bList[6], bList[7], bList[8]])
                dir2 = torch.cat([bList[9], bList[10], bList[11]])
                if ref_normal.matmul(ref_point.add(-center)) < 0:
                    ref_normal = -ref_normal
                newcenter = ref_normal.mul(2 * abs(torch.sum(ref_point.add(-center).mul(ref_normal)))).add(center)
                if ref_normal.matmul(dir1) < 0:
                    ref_normal = -ref_normal
                dir1 = dir1.add(ref_normal.mul(-2 * ref_normal.matmul(dir1)))
                if ref_normal.matmul(dir2) < 0:
                    ref_normal = -ref_normal
                dir2 = dir2.add(ref_normal.mul(-2 * ref_normal.matmul(dir2)))
                newbox = torch.cat([newcenter, dir0, dir1, dir2])
                reBoxes.append(newbox)

            boxes.extend(reBoxes)
            labels.extend(len(reBoxes) * [label])
    return boxes, labels


# %% md
# %%

def saveMats(boxes, syms, directory, suffix):
    # adjGen
    mat = torch.ones([4, 1]) * (-1)
    savemat(directory + '/adjGen.mat', {'adjGen': mat.numpy()})

    # boxes
    mat = torch.zeros([12, 30])
    for k in range(len(boxes)):
        mat[:, k] = boxes[k]
    key = 'boxes' + suffix
    savemat(directory + '/' + key + '.mat', {key: mat.numpy()})

    # syms
    mat = torch.ones([8, 30]) * (-1)
    for k in range(len(syms)):
        mat[:, k] = syms[k]
        if abs(mat[0, k] - 1) < 0.16:
            mat[0, k] = mat[0, k] - 1
        elif abs(mat[0, k]) < 0.16:
            mat[0, k] = mat[0, k] + 1
    key = 'syms' + suffix
    savemat(directory + '/' + key + '.mat', {key: mat.numpy()})

    # idx
    mat = torch.zeros([30, 1])
    for k in range(len(boxes)):
        mat[k] = k
    key = 'idx' + suffix
    savemat(directory + '/' + key + '.mat', {key: mat.numpy()})

    # adj
    mat = torch.zeros([30, 30])
    if len(boxes) == 3:
        mat[0, 2] = 1
        mat[2, 0] = 1
        mat[1, 2] = 1
        mat[2, 1] = 1
    key = 'adj' + suffix
    savemat(directory + '/' + key + '.mat', {key: mat.numpy()})


# %%

# for experiments where symmetry was ignored
def saveMatsNoSym(boxes, directory, suffix):
    # adjGen
    mat = torch.ones([4, 1]) * (-1)
    savemat(directory + '/adjGen.mat', {'adjGen': mat.numpy()})

    # boxes
    mat = torch.zeros([12, 30])
    for k in range(len(boxes)):
        mat[:, k] = boxes[k]
    key = 'boxes' + suffix
    savemat(directory + '/' + key + '.mat', {key: mat.numpy()})

    # syms
    mat = torch.ones([8, 30]) * (-1)
    for k in range(len(boxes)):
        mat[:, k] = 10
    key = 'syms' + suffix
    savemat(directory + '/' + key + '.mat', {key: mat.numpy()})

    # idx
    mat = torch.zeros([30, 1])
    for k in range(len(boxes)):
        mat[k] = np.random.randint(20)
    key = 'idx' + suffix
    savemat(directory + '/' + key + '.mat', {key: mat.numpy()})

    # adj
    mat = torch.zeros([30, 30])
    if len(boxes) == 3:
        mat[0, 2] = 1
        mat[2, 0] = 1
        mat[1, 2] = 1
        mat[2, 1] = 1
    key = 'adj' + suffix
    savemat(directory + '/' + key + '.mat', {key: mat.numpy()})


# %% md

## Take a chair from our dataset and format it for use with SCORES

# This function rearranges the boxes so that the ones with nontrivial symmetry parameters come first
# I've noticed this pattern in the SCORES input so just trying to be consistent with it


def reshuffle(boxes, syms, labels, objs):
    new_boxes = []
    new_syms = []
    new_labels = []
    new_objs = []
    for box, sym, label, obj in zip(boxes, syms, labels, objs):
        if sym[0] == 10.:
            new_boxes.append(box)
            new_syms.append(sym)
            new_labels.append(label)
            new_objs.append(obj)
        else:
            new_boxes = [box] + new_boxes
            new_syms = [sym] + new_syms
            new_labels = [label] + new_labels
            new_objs = [obj] + new_objs
    return new_boxes, new_syms, new_labels, new_objs


# load data from database
dir_syms = '../../partNet_syms/Chair'
dir_obj = '../../partNet_objs'
grassdataset = GRASSDataset(dir_syms,dir_obj,models_num=10, index=None)

new_tree1 = grassdataset[5]
new_tree2 = grassdataset[6]

allnewboxes1, allcopyBoxes1, allobjs = decode_structure(new_tree1.root)
showGenshape(allnewboxes1)
allnewboxes1 = unrotate_boxes(allnewboxes1)
allcopyBoxes1 = unrotate_boxes(allcopyBoxes1)

boxes1, syms1, labels1, objs1 = decode_boxes(new_tree1.root)

saveOBJ(allobjs,'testObj_A.OBJ' , renderBoxes2mesh(allnewboxes1,allcopyBoxes1,allobjs))

allnewboxes2, allcopyBoxes2, allobjs = decode_structure(new_tree2.root)
showGenshape(allnewboxes2)
allnewboxes2 = unrotate_boxes(allnewboxes2)
allcopyBoxes2 = unrotate_boxes(allcopyBoxes2)
showGenshape(allnewboxes2)
boxes2, syms2, labels2, objs2 = decode_boxes(new_tree2.root)

saveOBJ(allobjs, 'testObj_B.OBJ', renderBoxes2mesh(allnewboxes2,allcopyBoxes2,allobjs))

# select boxes with labels 0 and 1 from tree 1
ids = [i for i in range(len(labels1)) if labels1[i] in [3, 1]]
boxes_A = [boxes1[i] for i in ids]
syms_A = [syms1[i] for i in ids]
labels_A = [labels1[i] for i in ids]
objs_A = [objs1[i] for i in ids]
boxes_A, syms_A, labels_A, objs_A = reshuffle(boxes_A, syms_A, labels_A, objs_A)

# select boxes with labels 2 and 3 from tree 2
ids = [i for i in range(len(labels2)) if labels2[i] in [2, 0]]
boxes_B = [boxes2[i] for i in ids]
syms_B = [syms2[i] for i in ids]
labels_B = [labels2[i] for i in ids]
objs_B = [objs2[i] for i in ids]

boxes_B, syms_B, labels_B, objs_B = reshuffle(boxes_B, syms_B, labels_B, objs_B)

saveMats(boxes_A, syms_A, 'test_one_shape', 'A')
saveMats(boxes_B, syms_B, 'test_one_shape', 'B')


# %%
# Run scores on the new data
allTestData = SCORESTest('test_one_shape')
testFile = allTestData[0]
originalNodes = testFile.leves
input_boxes, copyBoxes, gtTypeBoxes, idxs = testVQContext.render_node_to_boxes(originalNodes)
input_boxes = unrotate_boxes(input_boxes)
copyBoxes = unrotate_boxes(copyBoxes)
showGenshape(input_boxes)

allObjs = []
for index in idxs:
    index = int(index.item())
    if index >= 10000:
        allObjs.append(objs_B[index-10000])
    else:
        allObjs.append(objs_A[index])

saveOBJ(allObjs,'sampled_before.OBJ' , renderBoxes2mesh(input_boxes,copyBoxes,allObjs))

mergeNetFix = torch.load('MergeNet_chair_demo_fix.pkl', map_location=lambda storage, loc: storage.cpu())
mergeNetFix = mergeNetFix.cpu()

# allBoxes = testVQContext.iterateKMergeTest(mergeNetFix, testFile)
iteration = 30
for i in range(iteration):
    nodes_ = testVQContext.oneIterMerge(mergeNetFix, testFile)
    
    # hack to prevent symmetry changes
    for node in nodes_:
        if node.sym[0] < 1:
            node.sym[0] = 1
    boxes_, _, _, idxs_ = testVQContext.render_node_to_boxes(originalNodes)

    allObjs_ = []
    for index in idxs_:
        index = int(index.item())
        if index >= 10000:
            allObjs_.append(objs_B[index - 10000])
        else:
            allObjs_.append(objs_A[index])

    copyBoxes_ = []
    for index in idxs_:
        index = int(index.item())
        index_loc = idxs.index(index)
        copyBoxes_.append(copyBoxes[index_loc])

    output_boxes = unrotate_boxes(boxes_)
    if i == iteration-1:
        showGenshape(output_boxes)
    # alignBoxAndRender(copyBoxes, output_boxes, gtTypeBoxes, allobjs, 'sampled_after_{}.OBJ'.format(i))
    print(len(allObjs_),len(allObjs_),len(copyBoxes_))
    saveOBJ(allObjs,'sampled_after_{}.OBJ'.format(i) , renderBoxes2mesh_new(output_boxes,copyBoxes_,allObjs_))


# %%
#
# print(len(allBoxes), 'output boxes:')
# for boxes in allBoxes:





