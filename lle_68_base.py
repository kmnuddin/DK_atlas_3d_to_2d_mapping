import mne
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding as LLE
from utils import save_json_result, save_2d_roi_map
from hyperopt import STATUS_OK

subject = 'fsaverage'

labels = mne.read_labels_from_annot('fsaverage')[0:68]

def get_all_vertices_dk_atlas_w_colors():
    pos = labels[0].pos.astype(np.float32)
    colors = np.full(len(pos), 1)
    for i,lbl in enumerate(labels[1:68]):
        pos = np.append(pos, lbl.pos.astype(np.float32), axis=0)
        c = np.full(len(lbl.pos), i+1)
        colors = np.append(colors, c)

    return pos, colors

def get_all_vertices_lh_w_color():
    labels_lh = [lbl for lbl in labels if lbl.hemi == 'lh']
    pos_lh = labels_lh[0].pos.astype(np.float32)
    for i,lbl in enumerate(labels_lh[1:34]):
        pos_lh = np.append(pos_lh, lbl.pos.astype(np.float32), axis=0)
        c = np.full(len(lbl.pos), i+1)
        colors_lh = np.append(colors_lh, c)

    return pos_lh, colors_lh

def get_all_vertices_rh_w_color():
    labels_rh = [lbl for lbl in labels if lbl.hemi == 'rh']
    pos_rh = labels_rh[0].pos.astype(np.float32)
    colors_rh = np.full(len(pos_rh), 1)
    for i,lbl in enumerate(labels_rh[1:34]):
        pos_lh = np.append(pos_rh, lbl.pos.astype(np.float32), axis=0)
        c = np.full(len(lbl.pos), i+1)
        colors_rh = np.append(colors_rh, c)

    return pos_rh, colors_rh


def get_centers_of_rois_xy(xy):
    start = 0
    end = 0
    centers = []
    for lbl in labels:
        n_vertices = len(lbl.pos)
        end = end + n_vertices
        x = xy[start:end,0].mean()
        y = xy[start:end,1].mean()
        centers.append((x,y))
        start = end
    return np.array(centers)

def avg_distance_between_center_of_masses(centers):
    distances = np.zeros((len(centers), len(centers)))
    for i in range(len(centers)):
        x, y = centers[i]

        for j in range(len(centers)):
            if i == j:
                continue

            x1, y1 = centers[j]

            d = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)

            distances[i,j] = d

    return np.mean(distances)

def lle(space):

    n_neighbors = int(space['n_neighbors'])
    method = space['method']

    vertices, colors = get_all_vertices_dk_atlas_w_colors()
    print(space)

    lle = LLE(n_neighbors=n_neighbors, n_components=2, method=method, neighbors_algorithm='auto')
    lle_xy = lle.fit_transform(vertices)

    centers = get_centers_of_rois_xy(lle_xy)

    avg_distance = avg_distance_between_center_of_masses(centers)

    model_name = 'lle_{}_{}'.format(method, avg_distance)

    result = {
        'loss': -avg_distance,
        'space': space,
        'status': STATUS_OK
    }

    save_json_result(model_name, result)
    save_2d_roi_map(lle_xy, colors, centers, model_name)

    return result
