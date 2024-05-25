from os.path import isfile
import math
import sys

POSE_TYPE_2D = "VERTEX_SE2"
POSE_TYPE_3D = "VERTEX_SE3:QUAT"
LANDMARK_TYPE_2D = "VERTEX_XY"
LANDMARK_TYPE_3D = "VERTEX_XYZ"

def save_pyfg_to_tum(pyfg_filepath: str, tum_filepath: str) -> None:
    """
    Parse a PyFactorGraph file to extract the factors and variables. Writes
    ground truth poses and landmark points to a file specified by tum_filepath.

    Args:
        filepath: The path to the PyFactorGraph file.

    Returns:
        None.

    Raises:
        ValueError: If the file does not exist.
        ValueError: If first line of file is not "VERTEX_SE2" or "VERTEX_SE3:QUAT".
        ValueError: If dimension is unknown.
    """

    if not isfile(pyfg_filepath):
        raise ValueError(f"File {pyfg_filepath} does not exist.")

    f_temp = open(pyfg_filepath, "r")

    def _get_dim_from_first_line(line: str) -> int:
        if line.startswith(POSE_TYPE_2D):
            return 2
        elif line.startswith(POSE_TYPE_3D):
            return 3
        else:
            raise ValueError(f"Unknown pose type {line}")
        
    dim = _get_dim_from_first_line(f_temp.readline())
    pose_state_dim = 3 if dim == 2 else 7
    f_temp.close()
    
    def _get_pose_from_line(line: str) -> str:
        line_parts = line.split(" ")
        pose_var_timestamp = float(line_parts[1])
        if (dim == 2):
            assert line_parts[0] == POSE_TYPE_2D
            assert len(line_parts) == 6
            x, y, theta = [float(x) for x in line_parts[-pose_state_dim:]]

            cr = 1 # math.cos(roll * 0.5)
            sr = 0 # math.sin(roll * 0.5)
            cp = 1 # math.cos(pitch * 0.5)
            sp = 0 # math.sin(pitch * 0.5)
            cy = math.cos(theta * 0.5) # math.cos(yaw * 0.5)
            sy = math.sin(theta * 0.5) # math.sin(yaw * 0.5)

            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy
            qw = cr * cp * cy + sr * sp * sy

            return f"{pose_var_timestamp} {x} {y} 0.0 {qx} {qy} {qz} {qw}"

        elif (dim == 3):
            assert line_parts[0] == POSE_TYPE_3D
            assert len(line_parts) == 10
            x, y, z, qx, qy, qz, qw = [float(x) for x in line_parts[-pose_state_dim:]]
            return f"{pose_var_timestamp} {x} {y} {z} {qx} {qy} {qz} {qw}"
        else:
            raise ValueError(f"Unknown dimension {dim}")
    
    #def _get_landmark_from_line(line: str) -> str:
    #    pass
    
    with open(pyfg_filepath, "r") as pyfg_file, open(tum_filepath, "w+") as tum_file:
        for line in pyfg_file:
            tokens = line.split(" ")
            # Ground truth poses and landmarks will only be stored in the below fields;
            # all other fields will be discarded
            if (tokens[0] == POSE_TYPE_2D or tokens[0] == POSE_TYPE_3D):
                pose_var = _get_pose_from_line(line)
                tum_file.write(f"{pose_var}\n")
            #elif (tokens[0] == LANDMARK_TYPE_2D or tokens[0] == LANDMARK_TYPE_3D):
            #    _get_landmark_from_line(line)
            else:
                continue

def main() -> None:
    save_pyfg_to_tum(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()