import numpy as np
from roboticstoolbox.robot.ERobot import ERobot

    
class PandaArm():
    def __init__(self, urdf_file): 
        
        self.robot = self.Panda(urdf_file)
        
    def get_joint_RT(self, joint_angle):
        
        assert joint_angle.shape[0] == 7


        link_idx_list = [0,1,2,3,4,5,6,7,9]
        # link 0,1,2,3,4,5,6,7, and hand
        R_list = []
        t_list = []
        

        for i in range(len(link_idx_list)):
            link_idx = link_idx_list[i]
            T = self.robot.fkine(joint_angle, end = self.robot.links[link_idx], start = self.robot.links[0])
            R_list.append(T.R)
            t_list.append(T.t)



        return np.array(R_list),np.array(t_list)
    
    def get_3d_keypoints(self, joint_angle):
        
        assert joint_angle.shape[0] == 7


        link_idx_list = [0,7]
        # link 0,7

        # keypoint for base
        base_keypoints = np.array([[0.0, 0.0, 0.0, 1.0],
                                    [0.06, 0.0, 0.0, 1.0],
                                    [0.0, 0.07, 0.0, 1.0],
                                    [0.0, 0.0, 0.1, 1.0],
                                    [-0.135, 0.0, 0.0, 1.0],
                                    [0.0, -0.07, 0.0, 1.0]])

        
        T = self.robot.fkine(joint_angle, end = self.robot.links[0], start = self.robot.links[0])
        base_points_h = np.array(T) @ base_keypoints.T
        base_points = base_points_h[:3,:].T

        # keypoint for end effector
        link6_keypoints = np.array([[0.0, 0.0, 0.0, 1.0],
                                    [0.0, 0.0, 0.05, 1.0],
                                    [0.0, 0.0, -0.125, 1.0]])
        T = self.robot.fkine(joint_angle, end = self.robot.links[6], start = self.robot.links[0])
        link6_points_h = np.array(T) @ link6_keypoints.T
        link6_points = link6_points_h[:3,:].T

        link7_keypoints = np.array([[0.0, 0.0, 0.0, 1.0],
                                 [0.0, 0.0, 0.05, 1.0],
                                [0.0, 0.0, -0.07, 1.0]])
        T = self.robot.fkine(joint_angle, end = self.robot.links[7], start = self.robot.links[0])
        link7_points_h = np.array(T) @ link7_keypoints.T
        link7_points = link7_points_h[:3,:].T

        return np.concatenate([base_points, link6_points, link7_points], axis = 0)
        
        
    class Panda(ERobot):
        """
        Class that imports a URDF model
        """

        def __init__(self, urdf_file):

            links, name, urdf_string, urdf_filepath = self.URDF_read(urdf_file)

            super().__init__(
                links,
                name=name,
                manufacturer="Franka",
                urdf_string=urdf_string,
                urdf_filepath=urdf_filepath,
                # gripper_links=elinks[9]
            )

    

