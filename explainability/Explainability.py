import lime 
import lime.lime_image
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions 
import numpy as np 
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt


def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        out.append(x)
    return np.vstack(out)

def get_explanation(model, images): 
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        images[0].astype('double'), 
        model.predict, 
        top_labels = 30, 
        hide_color = 0, 
        num_samples = 1000
    )
    
    return explanation 

def print_explanation_line(explanation): 
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=12, hide_rest=False)
    plt.imshow(mark_boundaries(temp.astype("uint8"), mask))

def print_heatmap(explanation): 
    #Select the same class explained on the figures above.
    ind =  explanation.top_labels[0]
    #Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
    #Plot. The visualization makes more sense if a symmetrical colorbar is used.
    plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
    plt.colorbar()