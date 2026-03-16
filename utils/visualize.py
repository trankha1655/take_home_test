
import numpy as np 
import tqdm
import cv2

def create_grid_image(image_list, shape, vis_grid=(10, 10)):
    count = 0 
    shape = (224, 224)  # Define the shape of the images 
    vis_image = np.zeros((shape[0] * vis_grid[0], shape[1] * vis_grid[1], 3), dtype=np.uint8)

    for iter, image in enumerate(tqdm.tqdm(image_list)):

        image = cv2.resize(image, shape)

        

        row = count // vis_grid[1]
        col = count % vis_grid[1]
        y_start = row * shape[0]
        y_end = y_start + shape[0]
        x_start = col * shape[1]
        x_end = x_start + shape[1]
        
        vis_image[y_start:y_end, x_start:x_end] = image.astype(np.uint8)  # Place the blended image in the grid
        count += 1
        if count >= vis_grid[0] * vis_grid[1]:
            return vis_image
        

    return vis_image


def visualize_probability_bars(image, class_names, probs):

    image = image.copy()
    h, w = image.shape[:2]
    scale = max(h, w) / 450 

    h_ = int(h / scale)
    w_ = int(w / scale)

    img_resize = cv2.resize(image, (int(w_), int(h_)))
    img = np.zeros((450, 450, 3))
    img[(450-h_)//2: (450-h_)//2 + h_, (450-w_)//2: (450-w_)//2 + w_, :] = img_resize 
    img = img.astype(np.uint8)

    bar_x = 20
    bar_y = 40
    bar_width = 150
    bar_height = 15
    gap = 17

    for i, (name, prob) in enumerate(zip(class_names, probs)):

        y = bar_y + i * gap

        # background bar
        cv2.rectangle(img,
                      (bar_x, y),
                      (bar_x + bar_width, y + bar_height),
                      (50, 50, 50), -1)

        # probability bar
        cv2.rectangle(img,
                      (bar_x, y),
                      (bar_x + int(bar_width * prob), y + bar_height),
                      (0, 255, 0), -1)

        label = f"{name}: {prob:.2f}"

        cv2.putText(img, label,
                    (bar_x + bar_width + 10, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255,255,255), 1)

    return img