import os
from bson import json_util
import json
import matplotlib.pyplot as plt

results_directory = 'results'
img_directory = '2d_cortical_imgs'

def save_json_result(model_name, result):
    """Save json to a directory and a filename."""
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    with open(os.path.join(results_directory, result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )


def load_json_result(best_result_name):
    """Load json from a path (directory + filename)."""
    result_path = os.path.join(results_directory, best_result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(
            f.read()
            # default=json_util.default,
            # separators=(',', ': ')
        )


def load_best_hyperspace():
    results = [
        f for f in list(sorted(os.listdir(results_directory))) if 'json' in f
    ]
    if len(results) == 0:
        return None

    best_result_name = results[-1]
    return load_json_result(best_result_name)["space"]

def save_2d_roi_map(lle, colors, centers, name):
    plt.figure()
    plt.scatter(lle[:,0], lle[:,1], c=colors, cmap=plt.cm.Spectral)

    for i,center in enumerate(centers):
        plt.annotate(i+1, (center[0],center[1]))

    plt.savefig(os.path.join(img_directory, '{}.png'.format(name)))

    plt.close()
