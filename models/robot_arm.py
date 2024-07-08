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
    
    def get_base_ee_RT(self, joint_angle):
        
        assert joint_angle.shape[0] == 7


        link_idx_list = [0,7]
        # link 0,7
        R_list = []
        t_list = []
        

        for i in range(len(link_idx_list)):
            link_idx = link_idx_list[i]
            T = self.robot.fkine(joint_angle, end = self.robot.links[link_idx], start = self.robot.links[0])
            R_list.append(T.R)
            t_list.append(T.t)



        return np.array(R_list),np.array(t_list)
        
        
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

    

