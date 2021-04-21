# helper function to find which direction best fits to x, y or z - used in rotate_boxes
def bestDir(dirs, lens, comp):
    largestComp = 0
    bestDirection = dirs[0]
    bestLen = lens[0]
    for k in range(len(dirs)):
        direction = dirs[k]
        if direction[comp] > largestComp:
            bestDirection = direction
            largestComp = direction[comp]
            bestLen = lens[k]
        elif -direction[comp] > largestComp:
            bestDirection = -direction
            largestComp = -direction[comp]
            bestLen = lens[k]
    return bestDirection, bestLen


# newest version of box rotation
def rotate_boxes(boxes, syms):
    new_boxes = torch.zeros(boxes.shape)

    for k in range(boxes.shape[0]):
        # rotate center vector
        new_boxes[k, 0] = boxes[k, 2]
        new_boxes[k, 1] = boxes[k, 1]
        new_boxes[k, 2] = -boxes[k, 0]

        # rotate principal directions
        new_boxes[k, 6] = boxes[k, 8]
        new_boxes[k, 7] = boxes[k, 7]
        new_boxes[k, 8] = -boxes[k, 6]

        new_boxes[k, 9] = boxes[k, 11]
        new_boxes[k, 10] = boxes[k, 10]
        new_boxes[k, 11] = -boxes[k, 9]

        # re-compute principal directions
        dir_1 = new_boxes[k, 6:9]
        dir_2 = new_boxes[k, 9:]
        dir_1 = dir_1 / np.linalg.norm(dir_1)
        dir_2 = dir_2 / np.linalg.norm(dir_2)
        dir_3 = np.cross(dir_1, dir_2)
        dir_3 = dir_3 / np.linalg.norm(dir_3)

        # realign axes
        x_dir, x_len = bestDir([dir_1, dir_2, dir_3], boxes[k, 3:6], 0)
        y_dir, y_len = bestDir([dir_1, dir_2, dir_3], boxes[k, 3:6], 1)
        z_dir, z_len = bestDir([dir_1, dir_2, dir_3], boxes[k, 3:6], 2)

        new_boxes[k, 3] = y_len
        new_boxes[k, 4] = z_len
        new_boxes[k, 5] = x_len

        new_boxes[k, 6:9] = torch.FloatTensor(y_dir)
        new_boxes[k, 9:12] = torch.FloatTensor(z_dir)

    # rotate syms
    new_syms = torch.zeros(syms.shape)
    for k in range(syms.shape[0]):
        new_syms[k, 0] = syms[k, 0]

        new_syms[k, 1] = syms[k, 3]
        new_syms[k, 2] = syms[k, 2]
        new_syms[k, 3] = -syms[k, 1]

        new_syms[k, 4] = syms[k, 6]
        new_syms[k, 5] = syms[k, 5]
        new_syms[k, 6] = -syms[k, 4]

        new_syms[k, 7] = syms[k, 7]
    return new_boxes, new_syms


# reverse box rotation (input 1 by 12 tensor)
def unrotate_box(box):
    new_box = torch.zeros([1, 12])
    # rotate center vector
    new_box[0, 0] = -box[0, 2]
    new_box[0, 1] = box[0, 1]
    new_box[0, 2] = box[0, 0]

    # rotate principal directions
    new_box[0, 6] = -box[0, 8]
    new_box[0, 7] = box[0, 7]
    new_box[0, 8] = box[0, 6]

    new_box[0, 9] = -box[0, 11]
    new_box[0, 10] = box[0, 10]
    new_box[0, 11] = box[0, 9]

    # re-compute principal directions
    dir_1 = new_box[0, 6:9]
    dir_2 = new_box[0, 9:]
    dir_1 = dir_1 / np.linalg.norm(dir_1)
    dir_2 = dir_2 / np.linalg.norm(dir_2)
    dir_3 = np.cross(dir_1, dir_2)
    dir_3 = dir_3 / np.linalg.norm(dir_3)

    # realign axes
    x_dir, x_len = bestDir([dir_1, dir_2, dir_3], box[0, 3:6], 0)
    y_dir, y_len = bestDir([dir_1, dir_2, dir_3], box[0, 3:6], 1)
    z_dir, z_len = bestDir([dir_1, dir_2, dir_3], box[0, 3:6], 2)

    new_box[0, 3] = y_len
    new_box[0, 4] = z_len
    new_box[0, 5] = x_len

    new_box[0, 6:9] = torch.FloatTensor(y_dir)
    new_box[0, 9:12] = torch.FloatTensor(z_dir)

    return new_box
