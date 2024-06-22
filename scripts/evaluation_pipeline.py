import sys
import argparse
import os
import shutil
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

# As adapted from: https://www.tutorialspoint.com/How-to-delete-all-files-in-a-directory-with-Python
def delete_subdir_contents(subdir_path: str) -> None:
    try:
        with os.scandir(subdir_path) as it:
            for entry in it:
                if entry.is_file():
                    os.remove(entry)
                else:
                    shutil.rmtree(entry)
    except FileNotFoundError:
        logger.warning(f"Directory {subdir_path} not found.")
    except OSError:
        logger.error(f"Error deleting contents of {subdir_path}.")


class EvaluationPipeline:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir

    def evaluate(self) -> None:
        logger.info(f"Evaluation Pipeline starting...")
        self.process_data_files()
        logger.info(f"Evaluation Pipeline finished.")

    def check_evo_tools(self) -> None:
        """
        Checks for evo, evo_traj, evo_ape, evo_rpe, and evo_res in PATH.

        Raises:
            FileNotFoundError: If any of the evo tools are not found in PATH.

        Returns:
            None
        """
        # *nix OS family: use which command

        evo_commands = ['evo', 'evo_traj', 'evo_ape', 'evo_rpe', 'evo_res']

        for evo_command in evo_commands:
            if (subprocess.call(['which', evo_command], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) != 0):
                raise FileNotFoundError(f"{evo_command} not found in PATH")

    def compare_cora_gt(self, cora_subdir: str, gt_subdir: str, evo_subdir: str) -> None:
        """
        Compare CORA-generated TUM files against ground truth TUM files.
        Temporarily using placeholder TUM files for CORA.

        Args:
            cora_subdir (str): Directory containing CORA-generated TUM files.
            gt_subdir (str): Directory containing ground truth TUM files.
            evo_subdir (str): Directory to save evo results.

        Raises:
            AssertionError: If the number of CORA TUM files are not equal to the number of ground truth TUM files.

        Returns:
            None
        """
        
        # Sort filenames lexicographically to ensure correct mapping
        cora_tum_file_list = sorted([f for f in os.listdir(cora_subdir) if f.endswith(".tum")])
        gt_tum_file_list = sorted([f for f in os.listdir(gt_subdir) if (f.endswith(".tum") or f.endswith(".txt"))])

        # Must map each CORA TUM file to a ground truth TUM file
        assert len(cora_tum_file_list) == len(gt_tum_file_list), "Number of CORA TUM files must match number of ground truth TUM files"

        cora_gt_dict = {}

        for cora_tum_file, gt_tum_file in zip(cora_tum_file_list, gt_tum_file_list):
            cora_gt_dict[cora_tum_file] = gt_tum_file

        if os.path.exists(os.path.join(evo_subdir, "cora")):
            delete_subdir_contents(os.path.join(evo_subdir, "cora"))

        cora_comparison_subdir = create_subdir(evo_subdir, "cora")

        for cora_tum_file, gt_tum_file in cora_gt_dict.items():
            cora_tum_file_path = os.path.join(cora_subdir, cora_tum_file)
            gt_tum_file_path = os.path.join(gt_subdir, gt_tum_file)
            evo_comparison_subdir = create_subdir(cora_comparison_subdir, f"{cora_tum_file}_vs_{gt_tum_file}")
            self.generate_evo_comparison(cora_tum_file_path, gt_tum_file_path, evo_comparison_subdir, use_gt=True)

    def compare_dcora_gt(self, dcora_subdir: str, gt_subdir: str, evo_subdir: str) -> None:
        """
        Compare DCORA-generated TUM files against ground truth TUM files.
        Temporarily using placeholder TUM files for DCORA.

        Args:
            dcora_subdir (str): Directory containing DCORA-generated TUM files.
            gt_subdir (str): Directory containing ground truth TUM files.
            evo_subdir (str): Directory to save evo results.

        Raises:
            AssertionError: If the number of DCORA TUM files are not equal to the number of ground truth TUM files.

        Returns:
            None
        """

        # Sort filenames lexicographically to ensure correct mapping
        dcora_tum_file_list = sorted([f for f in os.listdir(dcora_subdir) if f.endswith(".tum")])
        gt_tum_file_list = sorted([f for f in os.listdir(gt_subdir) if (f.endswith(".tum") or f.endswith(".txt"))])

        # Must map each DCORA TUM file to a ground truth TUM file
        assert len(dcora_tum_file_list) == len(gt_tum_file_list), "Number of DCORA TUM files must match number of ground truth TUM files"

        dcora_gt_dict = {}

        for dcora_tum_file, gt_tum_file in zip(dcora_tum_file_list, gt_tum_file_list):
            dcora_gt_dict[dcora_tum_file] = gt_tum_file

        # Prevents bug where evo commands would hang if comparisons already existed
        if os.path.exists(os.path.join(evo_subdir, "dcora")):
            delete_subdir_contents(os.path.join(evo_subdir, "dcora"))

        dcora_comparison_subdir = create_subdir(evo_subdir, "dcora")

        for dcora_tum_file, gt_tum_file in dcora_gt_dict.items():
            dcora_tum_file_path = os.path.join(dcora_subdir, dcora_tum_file)
            gt_tum_file_path = os.path.join(gt_subdir, gt_tum_file)
            evo_comparison_subdir = create_subdir(dcora_comparison_subdir, f"{dcora_tum_file}_vs_{gt_tum_file}")
            self.generate_evo_comparison(dcora_tum_file_path, gt_tum_file_path, evo_comparison_subdir, use_gt=True)

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

        # TODO: Create separate folders for each comparison between CORA/DCORA-generated TUM files and ground truth TUM files
        # For now, all output is dumped into the same evo_subdir

        evo_traj_command = ""

        if use_gt:
            evo_traj_command = f"evo_traj tum {tum_file_1} --ref={tum_file_2} -p --plot_mode=xz --save_plot {evo_subdir}/traj.png"
        else:
            evo_traj_command = f"evo_traj tum {tum_file_1} {tum_file_2} -p --plot_mode=xz --save_plot {evo_subdir}/traj.png"

        # To obtain png files of plots, use --save_plot flag instead of --save_results flag
        evo_ape_command = f"evo_ape tum {tum_file_1} {tum_file_2} -va --plot --plot_mode=xz --save_results {evo_subdir}/ape.zip --save_plot {evo_subdir}/ape.png"

        evo_rpe_command = f"evo_rpe tum {tum_file_1} {tum_file_2} -va --plot --plot_mode=xz --save_results {evo_subdir}/rpe.zip --save_plot {evo_subdir}/rpe.png"

        # Res will plot APE and save the statistics in a table
        evo_res_command = f"evo_res {evo_subdir}/*.zip --save_table {evo_subdir}/res_table.csv --save_plot {evo_subdir}/res.png"

        subprocess.run(evo_traj_command, shell=True, capture_output=True)
        subprocess.run(evo_ape_command, shell=True, capture_output=True)
        subprocess.run(evo_rpe_command, shell=True, capture_output=True)
        subprocess.run(evo_res_command, shell=True, capture_output=True)

        logger.info(f"{tum_file_1}: EVO comparison generated in {evo_subdir}.")

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

            # process data file to obtain ground truth
            save_robot_trajectories_to_tum_file(
                read_from_pyfg_file(data_file_path), ground_truth_subdir
            )

            # process data file with CORA
            # TODO(AT): create custom CORA executable that will operate on data_file and save its output to a CORA directory in data_file_out_dir

            # process data file with DCORA
            # TODO(AT): create custom DCORA executable that will operate on data_file and save its output to a DCORA directory in data_file_out_dir

            # compare CORA and DCORA against ground truth
            # TODO(JV): Use EVO for comparison. Consider using subprocess to call EVO from the command line.

            self.check_evo_tools()

            logger.info(f"{data_file}: Comparing CORA and DCORA against ground truth.")
        
            self.compare_cora_gt(cora_subdir, ground_truth_subdir, evo_subdir)
            self.compare_dcora_gt(dcora_subdir, ground_truth_subdir, evo_subdir)
            



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
