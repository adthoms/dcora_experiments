import sys
import argparse
import os
import logging
import subprocess
from py_factor_graph.io.pyfg_file import read_from_pyfg_file
from py_factor_graph.io.tum_file import save_robot_trajectories_to_tum_file
from py_factor_graph.utils.logging_utils import logger

logging.basicConfig(level=logging.INFO)


def create_subdir(dir: str, subdir_name: str) -> str:
    subdir_path = os.path.join(dir, subdir_name)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)
    return subdir_path


class EvaluationPipeline:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir

    def evaluate(self) -> None:
        logger.info(f"Evaluation Pipeline starting...")
        self.process_data_files()
        logger.info(f"Evaluation Pipeline finished.")

    def compare_cora_gt(self, cora_subdir: str, gt_subdir: str) -> None:
        """
        Compare CORA-generated TUM files against ground truth TUM files.
        Temporarily using placeholder TUM files for CORA.

        Args:
            cora_subdir (str): Directory containing CORA-generated TUM files.
            gt_subdir (str): Directory containing ground truth TUM files.

        Returns:
            None
        """
        pass

    def compare_dcora_gt(self, dcora_subdir: str, gt_subdir: str) -> None:
        """
        Compare DCORA-generated TUM files against ground truth TUM files.
        Temporarily using placeholder TUM files for DCORA.

        Args:
            dcora_subdir (str): Directory containing DCORA-generated TUM files.
            gt_subdir (str): Directory containing ground truth TUM files.

        Returns:
            None
        """
        pass

    def generate_evo_comparison(self, tum_file_1: str, tum_file_2: str, evo_subdir: str, use_gt: bool = True) -> None:
        """
        Run evo_traj, evo_ape, evo_rpe, and evo_res on inputted TUM files.

        Args:
            tum_file_1 (str): Filepath for the first TUM file.
            tum_file_2 (str): Filepath for the second TUM file.
            evo_subdir (str): Directory to save evo results.
            use_gt (bool, optional): Whether to treat tum_file_2 as ground truth. Defaults to True.

        Returns:
            None
        """
        pass

    def process_data_files(self) -> None:
        data_file_list = [f for f in os.listdir(self.data_dir) if f.endswith(".pyfg")]
        for data_file in data_file_list:
            # create separate directory for each data file
            data_file_out_dir = create_subdir(self.output_dir, data_file)

            # and add subdirectories for ground truth, cora, and dcora
            ground_truth_subdir = create_subdir(data_file_out_dir, "ground_truth")
            cora_subdir = create_subdir(data_file_out_dir, "cora")
            dcora_subdir = create_subdir(data_file_out_dir, "dcora")
            evo_subdir = create_subdir(data_file_out_dir, "evo")

            # get data file path
            data_file_path = os.path.join(self.data_dir, data_file)

            print("Saving ground truth")
            print("Printing true trajectories dict")
            # print(read_from_pyfg_file(data_file_path))
            print("Printed true trajectories dict")
            # process data file to obtain ground truth
            save_robot_trajectories_to_tum_file(
                read_from_pyfg_file(data_file_path), ground_truth_subdir
            )

            print("Saved ground truth to TUM file")

            # process data file with CORA
            # TODO(AT): create custom CORA executable that will operate on data_file and save its output to a CORA directory in data_file_out_dir

            # Placeholder for CORA TUM output
            cora_tum_output = os.path.join(self.data_dir, "cora_se2.tum")

            # process data file with DCORA
            # TODO(AT): create custom DCORA executable that will operate on data_file and save its output to a DCORA directory in data_file_out_dir

            # Placeholder for DCORA TUM output; will adjust to have one TUM output per agent
            dcora_tum_output = ""

            # compare CORA and DCORA against ground truth
            # TODO(JV): Use EVO for comparison. Consider using subprocess to call EVO from the command line.

            ground_truth_tum = os.path.join(ground_truth_subdir, "odom_gt_robot_A.txt")
            
            evo_traj_command = f"evo_traj tum {cora_tum_output} --ref={ground_truth_tum} -p --plot_mode=xz --save_plot {evo_subdir}/traj.png"

            # For CORA only, must implement DCORA
            # To obtain png files of plots, use --save_plot flag instead of --save_results flag
            evo_ape_command = f"evo_ape tum {cora_tum_output} {ground_truth_tum} -va --plot --plot_mode=xz --save_results {evo_subdir}/ape.zip --save_plot {evo_subdir}/ape.png"

            evo_rpe_command = f"evo_rpe tum {cora_tum_output} {ground_truth_tum} -va --plot --plot_mode=xz --save_results {evo_subdir}/rpe.zip --save_plot {evo_subdir}/rpe.png"

            # Res will plot APE and save the statistics in a table
            evo_res_command = f"evo_res {evo_subdir}/*.zip --save_table {evo_subdir}/res_table.csv --save_plot {evo_subdir}/res.png"

            subprocess.run(evo_traj_command, shell=True, capture_output=True)
            subprocess.run(evo_ape_command, shell=True, capture_output=True)
            subprocess.run(evo_rpe_command, shell=True, capture_output=True)
            out = subprocess.run(evo_res_command, shell=True, capture_output=True)
            print(out)
            



def main(args):
    parser = argparse.ArgumentParser(
        description="This script is used to compare CORA and DCORA on benchmark datasets obeying PyFG formatting."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="directory containing *.pfyg data files for comparison",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="directory containing TUM formatted trajectories for comparison",
    )

    # parse args and process
    args = parser.parse_args()
    evaluation_pipeline = EvaluationPipeline(args)
    evaluation_pipeline.evaluate()


if __name__ == "__main__":
    main(sys.argv[1:])
