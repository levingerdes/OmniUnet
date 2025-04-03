import math
import numpy as np
import matplotlib.pyplot as plt

class MergedImage():
    def __init__(self):
        ## Depth aligned to rgb to thermal extrinsics
        self.extrin_tras_x   = -0.029
        self.extrin_tras_y   = -0.0472
        self.extrin_tras_z   = 0.0180


    def alignThermalToDepth(self, DepthCamera_obj, ThermalCamera_obj, depth_matrix, thermal_matrix):
        depth_height,   depth_width     = DepthCamera_obj.height,   DepthCamera_obj.width
        thermal_height, thermal_width   = ThermalCamera_obj.height, ThermalCamera_obj.width

        aligned_thermal_matrix = np.zeros((depth_height, depth_width))
        for u_depth in range(0, depth_height):
            for v_depth in range(0, depth_width):

                # Direct intrinsics for Depth Image -> Depth Cam coordinates
                x_depth_cam, y_depth_cam, z_depth_cam = DepthCamera_obj.projectCameraCoord(depth_matrix, (v_depth, u_depth))
                
                # Direct extrinsics from Depth Cam (see is is from rgb sensor as it is aligned) to Thermal Cam
                x_thermal_cam  =  x_depth_cam + self.extrin_tras_x  
                y_thermal_cam  =  y_depth_cam + self.extrin_tras_y
                z_thermal_cam  =  z_depth_cam + self.extrin_tras_z
                
                # Inverse intrinsics from Thermal Cam coordinates to Thermal Images coordinates
                u_thermal, v_thermal, d_thermal  = ThermalCamera_obj.deprojectCameraCoord(x_thermal_cam, y_thermal_cam, z_thermal_cam)

                # Check if deprojected coordinates are inside thermal frame
                if(u_thermal <= thermal_width and v_thermal <= thermal_height and u_thermal >= 1 and v_thermal >= 1):
                    u_thermal_int = math.floor(u_thermal)
                    v_thermal_int = math.floor(v_thermal)

                    # Equivalence of Thermal Image to Depth Image
                    aligned_thermal_matrix[u_depth, v_depth] = thermal_matrix[v_thermal_int, u_thermal_int]

        return aligned_thermal_matrix


    def computeDEM(self, DepthCamera_obj, depth_matrix, resolution, camera_angle, distance_limit):
        x_w_list, y_w_list, z_w_list = [], [], []
        resolution_multiplier        = int(1 / resolution)

        # Creating list of all the real x, y, z coords for each image pixel
        for u in range(0, DepthCamera_obj.width):
            for v in range(0, DepthCamera_obj.height):
                x_w, y_w, z_w = DepthCamera_obj.obtainWorldCoord(depth_matrix, (u, v), camera_angle)

                # Remove farther points
                if(x_w < distance_limit and y_w < distance_limit):
                    x_w_list.append(x_w)
                    y_w_list.append(y_w)
                    z_w_list.append(z_w)

        # Finding max and min points for ploting DEM matrix
        x_offset   = np.abs(np.nanmin(x_w_list))
        DEM_width  = int((np.nanmax(x_w_list) + x_offset) * resolution_multiplier) + 1
        DEM_height = int((np.nanmax(y_w_list))            * resolution_multiplier) + 1

        # Initialize empty DEM
        DEM    = np.empty((DEM_width, DEM_height))
        DEM[:] = np.nan

        scaled_x_w_list, scaled_y_w_list, scaled_z_w_list = [], [], []
        for i in range(0, len(x_w_list)):
            x_w = x_w_list[i]
            y_w = y_w_list[i]
            z_w = z_w_list[i]
            
            if(not math.isnan(x_w) and not math.isnan(y_w) and not math.isnan(z_w)):
                x_w_scaled = int((x_w + x_offset) * resolution_multiplier) 
                y_w_scaled = int((y_w             * resolution_multiplier))
                
                # Z in original meters
                DEM[x_w_scaled][y_w_scaled] = z_w
                scaled_x_w_list.append(x_w_scaled)
                scaled_y_w_list.append(y_w_scaled)
                scaled_z_w_list.append(z_w)

        # Converting coordinates lists to np arrays
        x_DEM_array     = np.array(scaled_x_w_list)
        y_DEM_array     = np.array(scaled_y_w_list)
        z_DEM_array     = np.array(scaled_z_w_list)
        scaled_x_offset = x_offset * resolution_multiplier

        return DEM, x_DEM_array, y_DEM_array, z_DEM_array, scaled_x_offset

    def computeSegmentationDEM(self, DepthCamera_obj, depth_matrix, segmentation_matrix, resolution, camera_angle, distance_limit):
        x_w_list, y_w_list, z_w_list = [], [], []
        resolution_multiplier        = int(1 / resolution)

        # Segmentation world values for x_w and y_w
        segment_w_list   = []

        # Creating list of all the real x, y, z coords for each image pixel
        for u in range(0, DepthCamera_obj.width):
            for v in range(0, DepthCamera_obj.height):
                x_w, y_w, z_w = DepthCamera_obj.obtainWorldCoord(depth_matrix, (u, v), camera_angle)
                
                # Remove farther points
                if(x_w < distance_limit and y_w < distance_limit):
                    x_w_list.append(x_w)
                    y_w_list.append(y_w)
                    z_w_list.append(z_w)

                    # creating array for equivalent segmentation values of the images
                    segment_w_list.append(segmentation_matrix[v, u])

        # Finding max and min points for ploting DEM matrix
        x_offset   = np.abs(np.nanmin(x_w_list))
        DEM_width  = int((np.nanmax(x_w_list) + x_offset) * resolution_multiplier) + 1
        DEM_height = int((np.nanmax(y_w_list))            * resolution_multiplier) + 1

        # Initialize empty DEMs
        DEM            = np.empty((DEM_width, DEM_height))
        DEM_SEGMENT    = np.empty((DEM_width, DEM_height))
        DEM[:]         = np.nan
        DEM_SEGMENT[:] = np.nan

        scaled_x_w_list, scaled_y_w_list, scaled_z_w_list,  = [], [], []
        segmentation_values_list = []
        for i in range(0, len(x_w_list)):
            x_w = x_w_list[i]
            y_w = y_w_list[i]
            z_w = z_w_list[i]

            if(not math.isnan(x_w) and not math.isnan(y_w) and not math.isnan(z_w)):
                x_w_scaled = int((x_w + x_offset) * resolution_multiplier) 
                y_w_scaled = int((y_w             * resolution_multiplier))
                
                # DEM: Z in original meters
                DEM[x_w_scaled][y_w_scaled] = z_w
                scaled_x_w_list.append(x_w_scaled)
                scaled_y_w_list.append(y_w_scaled)
                scaled_z_w_list.append(z_w)

                # SEGMENT_DEM : 
                DEM_SEGMENT[x_w_scaled, y_w_scaled] = segment_w_list[i]
                segmentation_values_list.append(segment_w_list[i])

        
        # Converting coordinates lists to np arrays
        x_DEM_array       = np.array(scaled_x_w_list)
        y_DEM_array       = np.array(scaled_y_w_list)
        z_DEM_array       = np.array(scaled_z_w_list)
        segment_DEM_array = np.array(segmentation_values_list)
        scaled_x_offset   = x_offset * resolution_multiplier

        return DEM, DEM_SEGMENT, x_DEM_array, y_DEM_array, z_DEM_array, segment_DEM_array, scaled_x_offset



    def computeFusionDEM(self, DepthCamera_obj, depth_matrix, rgb_matrix, thermal_matrix, resolution, camera_angle, distance_limit):
        x_w_list, y_w_list, z_w_list = [], [], []
        resolution_multiplier        = int(1 / resolution)

        # RGB and thermal world values for x_w and y_w
        rgb_w_list, thermal_w_list   = [], []

        # Creating list of all the real x, y, z coords for each image pixel
        for u in range(0, DepthCamera_obj.width):
            for v in range(0, DepthCamera_obj.height):
                x_w, y_w, z_w = DepthCamera_obj.obtainWorldCoord(depth_matrix, (u, v), camera_angle)
                
                # Remove farther points
                if(x_w < distance_limit and y_w < distance_limit):
                    x_w_list.append(x_w)
                    y_w_list.append(y_w)
                    z_w_list.append(z_w)

                    # creating array for equivalent RGB and thermal values of the images
                    rgb_w_list.append((rgb_matrix[v, u, 0], rgb_matrix[v, u, 1], rgb_matrix[v, u, 2]))
                    thermal_w_list.append(thermal_matrix[v, u])

        # Finding max and min points for ploting DEM matrix
        x_offset   = np.abs(np.nanmin(x_w_list))
        DEM_width  = int((np.nanmax(x_w_list) + x_offset) * resolution_multiplier) + 1
        DEM_height = int((np.nanmax(y_w_list))            * resolution_multiplier) + 1

        # Initialize empty DEMs
        DEM            = np.empty((DEM_width, DEM_height))
        DEM_RGB        = np.empty((DEM_width, DEM_height, 3))
        DEM_THERMAL    = np.empty((DEM_width, DEM_height))
        DEM[:]         = np.nan
        DEM_RGB[:]     = np.nan
        DEM_THERMAL[:] = np.nan

        scaled_x_w_list, scaled_y_w_list, scaled_z_w_list,  = [], [], []
        rgb_values_list, thermal_values_list                = [], []
        for i in range(0, len(x_w_list)):
            x_w = x_w_list[i]
            y_w = y_w_list[i]
            z_w = z_w_list[i]

            if(not math.isnan(x_w) and not math.isnan(y_w) and not math.isnan(z_w)):
                x_w_scaled = int((x_w + x_offset) * resolution_multiplier) 
                y_w_scaled = int((y_w             * resolution_multiplier))
                
                # DEM: Z in original meters
                DEM[x_w_scaled][y_w_scaled] = z_w
                scaled_x_w_list.append(x_w_scaled)
                scaled_y_w_list.append(y_w_scaled)
                scaled_z_w_list.append(z_w)

                # RGB_DEM: Values of RGB between 0 and 1
                DEM_RGB[x_w_scaled, y_w_scaled, 0] =  rgb_w_list[i][0] / 255
                DEM_RGB[x_w_scaled, y_w_scaled, 1] =  rgb_w_list[i][1] / 255
                DEM_RGB[x_w_scaled, y_w_scaled, 2] =  rgb_w_list[i][2] / 255
                rgb_values_list.append(rgb_w_list[i])

                # THERMAL_DEM : 
                DEM_THERMAL[x_w_scaled, y_w_scaled] = thermal_w_list[i]
                thermal_values_list.append(thermal_w_list[i])

        
        # Converting coordinates lists to np arrays
        x_DEM_array       = np.array(scaled_x_w_list)
        y_DEM_array       = np.array(scaled_y_w_list)
        z_DEM_array       = np.array(scaled_z_w_list)
        rgb_DEM_array     = np.array(rgb_values_list) / 255 # Between 0 and 1 
        thermal_DEM_array = np.array(thermal_values_list)
        scaled_x_offset   = x_offset * resolution_multiplier

        return DEM, DEM_RGB, DEM_THERMAL, x_DEM_array, y_DEM_array, z_DEM_array, rgb_DEM_array, thermal_DEM_array, scaled_x_offset


    def calc3DDistance(self, v0, v1):
        euclidean_distance = math.sqrt((v0[0] - v1[0]) ** 2 + (v0[1] - v1[1]) ** 2 + (v0[2] -v1[2]) ** 2)
        return euclidean_distance


    def showDEM(self, DEM):
        fig = plt.figure()
        plt.imshow(DEM[:, :])
        # plt.ylim(260,160)
        # plt.xlim(20,120)
        plt.xlabel("y distance (cm)")
        plt.ylabel("x distance + xoffset (cm)")
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Elevation (cm)', rotation = 270, labelpad = 10)
        # plt.clim(0,-10)
        plt.close(fig)


    def showColor3Dscatter(self, x_array, y_array, z_array, rgb_array):
        fig  = plt.figure(figsize =(14, 9))
        ax   = plt.axes(projection ='3d')
        sctt = ax.scatter3D(x_array, y_array, z_array, marker = "o", s = 2.5, c = rgb_array)
        fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
        plt.show()


    def showMeshDEM(self, x_array, y_array, z_array):
        fig      = plt.figure(figsize  = (14, 9))
        ax       = plt.axes(projection = '3d')
        points2D = np.vstack([x_array, y_array]).T
        tri      = Delaunay(points2D)
        sctt     = ax.plot_trisurf(x_array, y_array, z_array, triangles = tri.simplices, cmap = plt.cm.Spectral)
        fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
        plt.show()


    def showAlignedOverlay(self, input_image, aligned_image):
        plt.figure()
        plt.imshow(input_image.astype(np.uint8), cmap  = 'gray')
        plt.imshow(aligned_image, alpha = 0.5)
        plt.show()


class CameraTransform():
    TYPE_REALSENSE = 0
    TYPE_THERMAL   = 1
    

    def __init__(self, camera_type, width, height, physical_height):
        
        if camera_type == self.TYPE_REALSENSE:
            # Command: s-enumerate-devices -c
            self.hfov            = 70.1
            self.vfov            = 43.15
            self.fx              = 912.04
            self.fy              = 910.34
            self.Cu              = 660.71   # exact half would be 640
            self.Cv              = 369.75   # exact half would be 360
            self.pixel_size      = 1.4e-6   # um
            self.focal_length    = 1.88e-3  # mm
            self.width           = width
            self.height          = height
            self.phys_h          = physical_height

        elif camera_type == self.TYPE_THERMAL:
            self.hfov            = 60
            self.vfov            = 45
            self.fx              = 555   # focal_length / pixel_size 
            self.fy              = 560   # focal_length / pixel_size 
            self.Cu              = 315      # exact half will be 320  
            self.Cv              = 240      # exact half will be 240  
            self.pixel_size      = 17e-6    # um from uncooled FPA detector
            self.focal_length    = 10.5e-3  # 60 FOV optics mm
            self.width           = width
            self.height          = height
            self.phys_h          = physical_height

        ## Depth aligned to rgb to thermal extrinsics
        self.extrin_tras_x   = -0.029
        self.extrin_tras_y   = -0.0472
        self.extrin_tras_z   = 0.0180


    def loadCsv(self, csv_file):
        file_array  = np.genfromtxt(csv_file, delimiter = ',')
        csv_matrix  = np.reshape(file_array, (self.height, self.width))
        
        return csv_matrix


    # This function is a direct transformation using (Image -> Camera) intrinsics
    def projectCameraCoord(self, depth_frame, px_coord):
        u = px_coord[0]
        v = px_coord[1]

        d = depth_frame[v,u]

        # Direct intrinsics transfrom
        x_p = (u - self.Cu) * d / self.fx
        y_p = (v - self.Cv) * d / self.fy
        
        # Return real coords in meters
        return (x_p, y_p, d)
    

    # This function is an inverse transformation using (Camera -> Image) intrinsics
    def deprojectCameraCoord(self, x_p, y_p, d):
        u = (x_p * self.fx) / d + self.Cu
        v = (y_p * self.fy) / d + self.Cv 

        return (u, v, d)


    # This function is an direct transformation using (Camera -> World) extrinsics
    def rotateRefSystem(self, coord_3d, camera_angle):
        angle_radians = math.radians(camera_angle)
        ref_rotation  = math.pi/2

        x_p = coord_3d[0]
        y_p = coord_3d[1]
        d   = coord_3d[2]

        # Left-hand reference system (meters)
        x_r =   x_p
        y_r =   y_p * math.cos(ref_rotation + angle_radians) + d * math.sin(ref_rotation + angle_radians)
        z_r = - y_p * math.sin(ref_rotation + angle_radians) + d * math.cos(ref_rotation + angle_radians) + self.phys_h

        return(x_r, y_r, z_r)
    

    # This function is an inverse transformation using (World -> camera) extrinsics
    def unrotateRefSystem(self, coord_3d, camera_angle):
        angle_radians = math.radians(camera_angle)
        ref_rotation  = math.pi/2

        x_r = coord_3d[0]
        y_r = coord_3d[1]
        z_r = coord_3d[2]

        # Left-hand reference system (meters)
        x_p = x_r
        y_p = y_r * math.cos(ref_rotation + angle_radians) - (z_r - self.phys_h) * math.sin(ref_rotation + angle_radians)
        z_p = y_r * math.sin(ref_rotation + angle_radians) + (z_r - self.phys_h) * math.cos(ref_rotation + angle_radians)

        return(x_p, y_p, z_p)


    def obtainWorldCoord(self, depth_frame, px_coord, camera_angle):
        # Applying direct intrinsics from image to Camera coords
        # (Image -> Camera)
        x_cam, y_cam, z_cam = self.projectCameraCoord(depth_frame,px_coord)

        # Applying direct extrinsics from Camera coords to world
        # (Camera -> World)
        x_w, y_w, z_w       = self.rotateRefSystem((x_cam, y_cam, z_cam), camera_angle)
        
        return (x_w, y_w, z_w)


    def obtainImageCoord(self, coord_3d, camera_angle):
        # Applying inverse extrinsics from World to camera coords
        # (World -> Camera)
        x_cam, y_cam, z_cam = self.unrotateRefSystem(coord_3d, camera_angle)

        # Applying inverse intrinsics from Camera to Image
        # (Camer -> World)
        u, v, d       = self.deprojectCameraCoord(x_cam, y_cam, z_cam)

        return (u, v, d)
