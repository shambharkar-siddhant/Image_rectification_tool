import cv2
import numpy as np
import tkinter as tk

class ImageTool:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.image_copy = self.image.copy()
        self.points = []
        self.window_name = "Image Tool"
        
    def run(self):
        self.create_window()
        
    def create_window(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse_click)
        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                self.image = self.image_copy.copy()
                self.points = []
            elif key == ord('q'):
                cv2.destroyAllWindows()
                break
        
        
    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            if len(self.points) == 1:
                cv2.circle(self.image, self.points[0], 2, (0, 0, 255), -1)
            elif len(self.points) == 2:
                cv2.circle(self.image, self.points[1], 2, (0, 0, 255), -1)
                length = np.sqrt((self.points[0][0] - self.points[1][0]) ** 2 +
                                 (self.points[0][1] - self.points[1][1]) ** 2)
                cv2.putText(self.image, f"Length: {length:.2f} pixels",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif len(self.points) == 3:
                self.extract_artwork()
    
    
    def matrix_rectification(arg,affine_rectified):
        # load the affine rectified artwork image
        # affine_rectified = cv2.imread('warped_image.jpg')

        # define the dimensions of the target rectangle in the metric rectified image
        target_width = 1000
        target_height = 800
        target_rect = np.array([[0, 0], [0, target_height], [target_width, target_height], [target_width, 0]], dtype=np.float32)

        # define a list to store the corner points
        points = []

        # define a function to capture mouse events
        def capture_points(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                if len(points) < 4:
                    points.append([x, y])
                    cv2.circle(affine_rectified, (x, y), 3, (0, 0, 255), -1)
                    cv2.imshow('Affine Rectified', affine_rectified)

        # display the affine rectified image and capture the corner points
        cv2.imshow('Affine Rectified', affine_rectified)
        cv2.setMouseCallback('Affine Rectified', capture_points)

        # wait until the user has selected all four corners
        while len(points) < 4:
            cv2.waitKey(1)

        # convert the corner points to a numpy array
        points = np.array(points, dtype=np.float32)

        # compute the homography matrix
        H, _ = cv2.findHomography(points, target_rect)

        # perform metric rectification
        metric_rectified = cv2.warpPerspective(affine_rectified, H, (target_width, target_height))

        # display the rectified image
        cv2.imshow('Metric Rectified', metric_rectified)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




    def affine_rectified(self,artwork):
        # load the artwork image
        # artwork = cv2.imread('artwork.jpg')

        # define the target rectangle
        target_rect = np.array([[0, 0], [0, artwork.shape[0]], [artwork.shape[1]*0.8, artwork.shape[0]]], dtype=np.float32)

        # create a window to display the artwork image
        cv2.namedWindow('Artwork')

        # define a callback function to get the corner points
        def get_points(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONUP and len(param['points']) < 4:
                param['points'].append((x, y))
                cv2.circle(param['img'], (x, y), 3, (0, 255, 0), -1)
                cv2.imshow('Artwork', param['img'])
            elif event == cv2.EVENT_LBUTTONDBLCLK and len(param['points']) == 4:
                param['done'] = True

        # get the corner points from the user
        points = []
        done = False
        while not done:
            img = artwork.copy()
            cv2.putText(img, 'Click on the corners of the artwork to select them and q to finish afterwards\n please select the points in counter clockwise fashion' , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('Artwork', img)
            cv2.setMouseCallback('Artwork', get_points, {'img': img, 'points': points, 'done': done})
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # convert the corner points to a numpy array
        points = np.array(points, dtype=np.float32)

        # compute the transformation matrix
        M = cv2.getAffineTransform(points[:3], target_rect)

        # perform affine rectification
        rectified = cv2.warpAffine(artwork, M, (int(artwork.shape[1]*0.8), artwork.shape[0]))

        
        # display the rectified image
        cv2.imshow('Rectified', rectified)

        self.matrix_rectification(rectified)

        cv2.waitKey(0)
        cv2.destroyAllWindows()



    def homographic(self,image):
        # Load the image and display it
        # image = cv2.imread('painting.jpg')
        cv2.imshow('Artwork', image)

        # Define the four corners of the original artwork
        original_corners = np.array([[100, 100], [100, 500], [500, 500], [500, 100]], dtype=np.float32)

        # Define the four corners of the target rectangle
        target_corners = np.array([[100, 100], [100, 700], [700, 700], [700, 100]], dtype=np.float32)

        # Compute the homography matrix using the corresponding points
        M, _ = cv2.findHomography(original_corners, target_corners)

        # Warp the artwork to the target rectangle using the homography matrix
        warped_image = cv2.warpPerspective(image, M, (800, 800))

        
        # Display the warped artwork
        cv2.imshow('Warped Artwork', warped_image)

        self.affine_rectified(image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
    def extract_artwork(self):
        # load the image
        img = cv2.imread('painting.jpg')

        # convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # perform edge detection
        edges = cv2.Canny(gray, 100, 200)

        # find the contours in the image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # get the bounding box of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # extract the artwork from the image
        artwork = img[y:y+h, x:x+w]

        self.homographic(artwork)

        # display the artwork
        # cv2.imshow('Artwork', artwork)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    

if __name__ == '__main__':
    tool = ImageTool("painting.jpg")
    tool.run()
