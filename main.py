import GCPfunctionDefinations as gcp
import numpy as np
import cv2
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """Main Body"""

    # Read Original Image as Colored
    path = gcp.file_browse()
    img = cv2.imread(path, 1)

    # Delete exexting output folder and create a new folder
    if os.path.isdir("output"):
        shutil.rmtree("output")
    if not os.path.isdir("output"):
        os.mkdir("output")

    # Read Template Image
    template = cv2.imread("template.JPG", 0)
    w = template.shape[1]
    h = template.shape[0]

    # Grey Scale Original Image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Deciding Threshold
    px_max = np.amax(img_gray)
    lower_limit = int(px_max * 0.8)
    upper_limit = px_max

    # Thresholding the image
    orig_img = gcp.get_filteredimage(img, lower_limit, upper_limit)

    # DataFrame to store the GCP location
    location = pd.DataFrame(columns={'Name', 'loc'})
    name = pd.Series(path)
    X = []
    Y = []

    # Slicing the original image into 1000X1000 pixel images
    for i in range(0, img.shape[0], 1000):
        for j in range(0, img.shape[1], 1000):

            img_block = img[i:i+1000, j:j+1000]
            x_start = j
            y_start = i
            filt_block = gcp.get_filteredimage(img_block, lower_limit, upper_limit)
            # gcp.plot_image(filt_block, "Threshold Image")

            # Template Matching
            block_mean = np.mean(filt_block)
            print("block mean:", block_mean)
            if block_mean > 1:
                thresh = 0.7
            elif block_mean < 0.1:
                thresh = 0.2
            else:
                thresh = 0.5

            if block_mean > 0.02:
                loc = gcp.template_matching(filt_block, template, thresh)

                # for pt in zip(*loc[::-1]):
                #     cv2.rectangle(img_block, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
                # gcp.plot_image(img_block)

                # Feature Matching
                matches, _kp1, _kp2 = gcp.feature_matching(filt_block, template)

                # Extract Pixel positions for matched features
                kp1_pos, kp2_pos, _ = gcp.pixel_positions(_kp1, _kp2, matches)

                # Pixel position refinement
                refined_points, refined_keypoints = gcp.position_refinement(loc, kp1_pos)

                # Drawing Square on the Detected Positions
                _tol = 3
                for point in range(0, len(refined_points), 1):
                    pt = refined_points[point]
                    print(pt)
                    pt_key = refined_keypoints[point]

                    temp_img = filt_block[pt[1]:(pt[1] + h), pt[0]:(pt[0] + w)]
                    average_color = [temp_img[:, :, i].mean() for i in range(temp_img.shape[-1])]
                    c1 = int(average_color[0])
                    c2 = int(average_color[1])
                    c3 = int(average_color[2])
                    print(c1, c2, c3)
                    if np.abs(c1-c2) <= _tol and np.abs(c2-c3) <= _tol and np.abs(c1-c3) <= _tol:
                        if np.abs(np.mean(temp_img) - np.mean(template)) < 30 or c1 == c2 == c3:
                            cv2.rectangle(img_block, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
                            X.append((pt_key[0] + x_start, pt_key[1] + y_start))
                            print(pt_key)

                # Save the Final Image
                # gcp.plot_image(img_block, "Detected Locations")
                cv2.imwrite("output/block("+str(i)+","+str(j)+").JPG", img_block)

    # Write the DataFrame into CSV
    location['Name'] = name
    location['loc'] = [X]
    location.to_csv("output/location.csv", sep=",")
    plt.imshow(img)
    plt.show()
