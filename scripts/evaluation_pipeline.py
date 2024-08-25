import argparse
import copy
import csv
import os
import shutil
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from evo.core import metrics
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.tools.plot import PlotMode, prepare_axis, traj, set_aspect_equal
from py_factor_graph.io.pyfg_file import read_from_pyfg_file
from py_factor_graph.io.tum_file import save_robot_trajectories_to_tum_file
from py_factor_graph.utils.logging_utils import logger

FONT_SIZE = 10
DPI = 1200

COMBINED_TRAJ_LEGEND_OFFSET = (1.15, -0.15)
COMBINED_TRAJ_LEGEND_LOC = "lower center"


def create_subdir(dir: str, subdir_name: str) -> str:
    subdir_path = os.path.join(dir, subdir_name)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)
    return subdir_path


def is_dir_empty(dir: str) -> bool:
    return not any(os.scandir(dir))


def delete_subdir_contents(subdir_path: str) -> None:
    logger.info(f"Deleting contents of {subdir_path} ...")
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


def get_sorted_file_list(
    subdir_path: str, file_extensions=[".txt", ".tum"]
) -> List[str]:
    file_list = [
        os.path.join(subdir_path, f)
        for f in os.listdir(subdir_path)
        if any(f.endswith(ext) for ext in file_extensions)
    ]
    return sorted(file_list)


def get_file_name_from_path(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def append_stats_to_csv(stats_dict: Dict[str, str], csv_file: str) -> None:
    with open(csv_file, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(stats_dict.keys()))
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(stats_dict)


def align_trajectories(
    tum_traj_est_file: str,
    evo_traj_ref: PoseTrajectory3D,
    correct_scale: bool = False,
    correct_only_scale: bool = False,
):
    evo_traj_est = file_interface.read_tum_trajectory_file(tum_traj_est_file)
    evo_traj_est_aligned = copy.deepcopy(evo_traj_est)
    evo_traj_est_aligned.align(
        evo_traj_ref, correct_scale=correct_scale, correct_only_scale=correct_only_scale
    )
    return evo_traj_est_aligned


def get_plot_mode_by_traj(cora_traj: PoseTrajectory3D, dcora_traj: PoseTrajectory3D, gt_traj: PoseTrajectory3D) -> PlotMode:
    isTraj2D = lambda traj: np.all(traj.positions_xyz[:, -1] == 0)
    if isTraj2D(cora_traj) and isTraj2D(dcora_traj) and isTraj2D(gt_traj):
        return PlotMode.xy
    else:
        return PlotMode.xyz


def plot_trajectories(
    cora_traj: PoseTrajectory3D,
    dcora_traj: PoseTrajectory3D,
    gt_traj: PoseTrajectory3D,
    agent_subdir: str,
):
    fig = plt.figure()
    trajectories = {"CORA": cora_traj, "DCORA": dcora_traj, "Ground Truth": gt_traj}
    traj_colors = ["red", "blue", "black"]

    # check for 2D or 3D plot
    plot_mode = get_plot_mode_by_traj(cora_traj, dcora_traj, gt_traj)

    # plot trajectories on aingle axis
    ax = prepare_axis(fig, plot_mode)
    for idx, (name, traj_path) in enumerate(trajectories.items()):
        line_style = "--" if name == "Ground Truth" else "-"
        traj(ax, plot_mode, traj_path, line_style, traj_colors[idx], name)

    output_traj_plot(ax, "traj", agent_subdir)


def add_traj_to_plot(
    ax,
    plot_mode: PlotMode,
    cora_traj: PoseTrajectory3D,
    dcora_traj: PoseTrajectory3D,
    gt_traj: PoseTrajectory3D,
):
    trajectories = {f"CORA": cora_traj, f"DCORA": dcora_traj, f"Ground Truth": gt_traj}
    traj_colors = ["red", "blue", "black"]

    # plot trajectories
    for idx, (name, traj_path) in enumerate(trajectories.items()):
        line_style = "--" if name == "Ground Truth" else "-"
        traj(ax, plot_mode, traj_path, line_style, traj_colors[idx], name)


def output_traj_plot(
    ax,
    name: str,
    output_subdir: str,
    legend_offset: Tuple[float, float] = None,
    legend_loc: str = None,
    override_legend: bool = False,
):
    # Set axis to equal size
    set_aspect_equal(ax)

    # Set background colors
    ax.set_facecolor("white")

    # Combining trajectory plots also combines legends; override to use custom legend
    if override_legend:
        cora_line = Line2D([0], [0], color="red", linestyle="-", label="CORA")
        dcora_line = Line2D([0], [0], color="blue", linestyle="-", label="DCORA")
        gt_line = Line2D([0], [0], color="black", linestyle="--", label="Ground Truth")
        ax.legend(handles=[cora_line, dcora_line, gt_line], facecolor="white", loc=legend_loc, bbox_to_anchor=legend_offset, ncol=3)
    else:
        ax.legend(facecolor="white", loc=legend_loc, bbox_to_anchor=legend_offset)

    # Set edge colors
    ax.spines["top"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    # Set grid colors
    ax.grid(color="gray")

    # Set labels
    ax.set_xlabel("x (m)", fontsize=FONT_SIZE)
    ax.set_ylabel("y (m)", fontsize=FONT_SIZE)

    # save figure
    plt.tight_layout() # Ensure legend is not cut off if offset is used
    plt.savefig(os.path.join(output_subdir, f"{name}.png"), dpi=DPI)
    plt.close()


def calculate_stats(
    traj_pair_list: List[Tuple[PoseTrajectory3D, PoseTrajectory3D]],
    algorithm_name_list: List[str],
    agent_subdir: str,
) -> None:
    assert len(traj_pair_list) == len(algorithm_name_list), logger.critical(
        "The number of data pairs must match the number of algorithms!"
    )
    ape_trans_stats_csv_file = os.path.join(agent_subdir, f"ape_trans_stats.csv")
    ape_rot_stats_csv_file = os.path.join(agent_subdir, f"ape_rot_stats.csv")
    rpe_trans_stats_csv_file = os.path.join(agent_subdir, f"rpe_trans_stats.csv")
    rpe_rot_stats_csv_file = os.path.join(agent_subdir, f"rpe_rot_stats.csv")
    ape_trans_stats_dict = dict()
    ape_rot_stats_dict = dict()
    rpe_trans_stats_dict = dict()
    rpe_rot_stats_dict = dict()

    for i, data_pair in enumerate(traj_pair_list):

        # set algorithm name
        algorithm_name = algorithm_name_list[i]
        ape_trans_stats_dict["alg"] = f"{algorithm_name}"
        ape_rot_stats_dict["alg"] = f"{algorithm_name}"
        rpe_trans_stats_dict["alg"] = f"{algorithm_name}"
        rpe_rot_stats_dict["alg"] = f"{algorithm_name}"

        # calculate APE (trans, rot)
        ape_trans_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_trans_metric.process_data(data_pair)
        ape_trans_stats_dict.update(ape_trans_metric.get_all_statistics())
        ape_rot_metric = metrics.APE(metrics.PoseRelation.rotation_part)
        ape_rot_metric.process_data(data_pair)
        ape_rot_stats_dict.update(ape_rot_metric.get_all_statistics())

        # calculate RPE (trans, rot)
        rpe_trans_metric = metrics.RPE(
            pose_relation=metrics.PoseRelation.translation_part,
            all_pairs=True,  # use all pose pairs
        )
        rpe_trans_metric.process_data(data_pair)
        rpe_trans_stats_dict.update(rpe_trans_metric.get_all_statistics())
        rpe_rot_metric = metrics.RPE(
            pose_relation=metrics.PoseRelation.rotation_part,
            all_pairs=True,  # use all pose pairs
        )
        rpe_rot_metric.process_data(data_pair)
        rpe_rot_stats_dict.update(rpe_rot_metric.get_all_statistics())

        # save to csv file
        append_stats_to_csv(ape_trans_stats_dict, ape_trans_stats_csv_file)
        append_stats_to_csv(ape_rot_stats_dict, ape_rot_stats_csv_file)
        append_stats_to_csv(rpe_trans_stats_dict, rpe_trans_stats_csv_file)
        append_stats_to_csv(rpe_rot_stats_dict, rpe_rot_stats_csv_file)

def calculate_combined_stats(
        cora_tum_file_list: List[str],
        dcora_tum_file_list: List[str],
        gt_tum_file_list: List[str],
        output_subdir: str,
) -> None:
    """Calculate weighted average of all agents' statistics; weight corresponds to the number of poses each agent has.

    Args:
        cora_tum_file_list (List[str]): List of CORA TUM files
        dcora_tum_file_list (List[str]): List of DCORA TUM files
        gt_tum_file_list (List[str]): List of Ground Truth TUM files
    
    Returns:
        None
    """

    num_cora_trajectories = 0
    num_dcora_trajectories = 0
    # get total number of CORA/DCORA poses
    for cora_tum_file, dcora_tum_file, gt_tum_file in zip(
        cora_tum_file_list, dcora_tum_file_list, gt_tum_file_list
    ):
        # align trajectories
        gt_traj = file_interface.read_tum_trajectory_file(gt_tum_file)
        cora_traj_aligned = align_trajectories(cora_tum_file, gt_traj)
        dcora_traj_aligned = align_trajectories(dcora_tum_file, gt_traj)

        num_cora_trajectories += cora_traj_aligned.num_poses
        num_dcora_trajectories += dcora_traj_aligned.num_poses

    # calculate weighted average of statistics
    algorithm_name_list = ["cora", "dcora"]

    ape_trans_stats_csv_file = os.path.join(output_subdir, f"ape_trans_stats.csv")
    ape_rot_stats_csv_file = os.path.join(output_subdir, f"ape_rot_stats.csv")
    rpe_trans_stats_csv_file = os.path.join(output_subdir, f"rpe_trans_stats.csv")
    rpe_rot_stats_csv_file = os.path.join(output_subdir, f"rpe_rot_stats.csv")

    ape_trans_metric_dict = dict()
    ape_rot_metric_dict = dict()
    rpe_trans_metric_dict = dict()
    rpe_rot_metric_dict = dict()

    for algorithm_name in algorithm_name_list:
        ape_trans_metric_dict["alg"] = f"{algorithm_name}"
        ape_rot_metric_dict["alg"] = f"{algorithm_name}"
        rpe_trans_metric_dict["alg"] = f"{algorithm_name}"
        rpe_rot_metric_dict["alg"] = f"{algorithm_name}"

        for cora_tum_file, dcora_tum_file, gt_tum_file in zip(
            cora_tum_file_list, dcora_tum_file_list, gt_tum_file_list
        ):
            # align trajectories
            gt_traj = file_interface.read_tum_trajectory_file(gt_tum_file)
            cora_traj_aligned = align_trajectories(cora_tum_file, gt_traj)
            dcora_traj_aligned = align_trajectories(dcora_tum_file, gt_traj)

            # calculate stats
            traj_pair = None
            agent_weight = 0.0

            if algorithm_name == "cora":
                traj_pair = (gt_traj, cora_traj_aligned)
                agent_weight = cora_traj_aligned.num_poses / num_cora_trajectories
            else:
                traj_pair = (gt_traj, dcora_traj_aligned)
                agent_weight = dcora_traj_aligned.num_poses / num_dcora_trajectories

            # calculate APE (trans, rot)
            ape_trans_metric = metrics.APE(metrics.PoseRelation.translation_part)
            ape_trans_metric.process_data(traj_pair)
            # ape_trans_stats_dict.update(ape_trans_metric.get_all_statistics())
            ape_rot_metric = metrics.APE(metrics.PoseRelation.rotation_part)
            ape_rot_metric.process_data(traj_pair)
            # ape_rot_stats_dict.update(ape_rot_metric.get_all_statistics())

            # calculate RPE (trans, rot)
            rpe_trans_metric = metrics.RPE(
                pose_relation=metrics.PoseRelation.translation_part,
                all_pairs=True,  # use all pose pairs
            )
            rpe_trans_metric.process_data(traj_pair)
            # rpe_trans_stats_dict.update(rpe_trans_metric.get_all_statistics())
            rpe_rot_metric = metrics.RPE(
                pose_relation=metrics.PoseRelation.rotation_part,
                all_pairs=True,  # use all pose pairs
            )
            rpe_rot_metric.process_data(traj_pair)
            # rpe_rot_stats_dict.update(rpe_rot_metric.get_all_statistics())

            for stat, value in ape_trans_metric.get_all_statistics().items():
                ape_trans_metric_dict.setdefault(stat, 0)
                ape_trans_metric_dict[stat] += value * agent_weight
            
            for stat, value in ape_rot_metric.get_all_statistics().items():
                ape_rot_metric_dict.setdefault(stat, 0)
                ape_rot_metric_dict[stat] += value * agent_weight
            
            for stat, value in rpe_trans_metric.get_all_statistics().items():
                rpe_trans_metric_dict.setdefault(stat, 0)
                rpe_trans_metric_dict[stat] += value * agent_weight
            
            for stat, value in rpe_rot_metric.get_all_statistics().items():
                rpe_rot_metric_dict.setdefault(stat, 0)
                rpe_rot_metric_dict[stat] += value * agent_weight
        append_stats_to_csv(ape_trans_metric_dict, ape_trans_stats_csv_file)
        append_stats_to_csv(ape_rot_metric_dict, ape_rot_stats_csv_file)
        append_stats_to_csv(rpe_trans_metric_dict, rpe_trans_stats_csv_file)
        append_stats_to_csv(rpe_rot_metric_dict, rpe_rot_stats_csv_file)

        # Clear dictionaries for next algorithm
        ape_trans_metric_dict = dict()
        ape_rot_metric_dict = dict()
        rpe_trans_metric_dict = dict()
        rpe_rot_metric_dict = dict()


class EvaluationPipeline:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.output_dir = args.output_dir
        self.override_cora_results = args.override_cora_results
        self.override_dcora_results = args.override_dcora_results
        self.override_ground_truth_results = args.override_ground_truth_results
        self.override_evo_results = args.override_evo_results

    def _cora_filter(self, data_file_path: str, cora_subdir: str) -> None:
        # TODO(AT): create custom CORA executable that will operate on data_file and save its output to cora_subdir
        pass

    def _dcora_filter(self, data_file_path: str, dcora_subdir: str) -> None:
        # TODO(AT): create custom DCORA executable that will operate on data_file and save its output to dcora_subdir
        pass

    def _ground_truth_filter(
        self, data_file_path: str, ground_truth_subdir: str, use_pose_idx: bool = True
    ) -> None:
        save_robot_trajectories_to_tum_file(
            read_from_pyfg_file(data_file_path),
            ground_truth_subdir,
            use_pose_idx=use_pose_idx,
        )

    def _evo_filter(
        self,
        cora_subdir: str,
        dcora_subdir: str,
        ground_truth_subdir: str,
        evo_subdir: str,
    ) -> None:
        # get sorted file lists
        cora_tum_file_list = get_sorted_file_list(cora_subdir)
        dcora_tum_file_list = get_sorted_file_list(dcora_subdir)
        gt_tum_file_list = get_sorted_file_list(ground_truth_subdir)
        assert len(cora_tum_file_list) == len(gt_tum_file_list), logger.critical(
            "Number of CORA TUM files must match number of ground truth TUM files!"
        )
        assert len(dcora_tum_file_list) == len(gt_tum_file_list), logger.critical(
            "Number of DCORA TUM files must match number of ground truth TUM files!"
        )

        # plot of combined trajectories
        ax_combined_traj = plt.figure().add_subplot(111)

        # apply filter to each agent
        for cora_tum_file, dcora_tum_file, gt_tum_file in zip(
            cora_tum_file_list, dcora_tum_file_list, gt_tum_file_list
        ):
            # check that all files belong to same agent
            agent_id = get_file_name_from_path(gt_tum_file)[-1]
            cora_agent_id = get_file_name_from_path(cora_tum_file)[-1]
            dcora_agent_id = get_file_name_from_path(dcora_tum_file)[-1]
            assert cora_agent_id == agent_id, logger.critical(
                f"{cora_tum_file} and {gt_tum_file} do not belong to same agent!"
            )
            assert dcora_agent_id == agent_id, logger.critical(
                f"{dcora_tum_file} and {gt_tum_file} do not belong to same agent!"
            )

            # save results in agent subdirectory
            agent_subdir = create_subdir(evo_subdir, f"{agent_id}")
            logger.info(f"Saving agent {agent_id} results to {agent_subdir} ...")

            # align trajectories
            gt_traj = file_interface.read_tum_trajectory_file(gt_tum_file)
            cora_traj_aligned = align_trajectories(cora_tum_file, gt_traj)
            dcora_traj_aligned = align_trajectories(dcora_tum_file, gt_traj)

            # plot trajectories
            plot_trajectories(
                cora_traj_aligned, dcora_traj_aligned, gt_traj, agent_subdir
            )

            # Add trajectories to combined plot
            plot_mode = get_plot_mode_by_traj(cora_traj_aligned, dcora_traj_aligned, gt_traj)
            add_traj_to_plot(
                ax_combined_traj,
                plot_mode,
                cora_traj_aligned,
                dcora_traj_aligned,
                gt_traj,
            )

            # calculate stats
            # TODO(JV): Create combined statistics aggregated for the whole dataset
            traj_pair_list = [
                (gt_traj, cora_traj_aligned),
                (gt_traj, dcora_traj_aligned),
            ]
            algorithm_name_list = ["cora", "dcora"]
            calculate_stats(traj_pair_list, algorithm_name_list, agent_subdir)

        combined_subdir = create_subdir(evo_subdir, "combined")
        logger.info(f"Saving combined trajectory output and weighted average statistics to {combined_subdir} ...")
        output_traj_plot(
            ax_combined_traj, 
            "traj_combined", 
            combined_subdir,
            override_legend=True,
            # legend_loc=COMBINED_TRAJ_LEGEND_LOC, 
            legend_offset=COMBINED_TRAJ_LEGEND_OFFSET,
        )

        calculate_combined_stats(
            cora_tum_file_list, dcora_tum_file_list, gt_tum_file_list, combined_subdir
        )

    def evaluate(self) -> None:
        logger.info("Starting evaluation pipeline...")
        data_file_list = [f for f in os.listdir(self.data_dir) if f.endswith(".pyfg")]
        for data_file in data_file_list:
            # get data file
            data_file_path = os.path.join(self.data_dir, data_file)

            # set output directory for data file
            data_file_out_dir = create_subdir(
                self.output_dir, get_file_name_from_path(data_file)
            )

            # set subdirectories of output directory
            cora_subdir = create_subdir(data_file_out_dir, "cora")
            dcora_subdir = create_subdir(data_file_out_dir, "dcora")
            ground_truth_subdir = create_subdir(data_file_out_dir, "ground_truth")
            evo_subdir = create_subdir(data_file_out_dir, "evo")

            # clear subdirectories
            if self.override_cora_results:
                delete_subdir_contents(cora_subdir)
            if self.override_dcora_results:
                delete_subdir_contents(dcora_subdir)
            if self.override_ground_truth_results:
                delete_subdir_contents(ground_truth_subdir)
            if self.override_evo_results:
                delete_subdir_contents(evo_subdir)

            # apply filters
            if is_dir_empty(ground_truth_subdir):
                logger.info(
                    f"Applying Ground Truth filter to data file: {data_file} ..."
                )
                self._ground_truth_filter(data_file_path, ground_truth_subdir)
            else:
                logger.info(
                    f"Ground Truth results exist for data file: {data_file}. Skipping Ground Truth filter."
                )

            if is_dir_empty(cora_subdir):
                logger.info(f"Applying CORA filter to data file: {data_file} ...")
                self._cora_filter(data_file_path, cora_subdir)
            else:
                logger.info(
                    f"CORA results exist for data file: {data_file}. Skipping CORA filter."
                )

            if is_dir_empty(dcora_subdir):
                logger.info(f"Applying DCORA filter to data file: {data_file} ...")
                self._dcora_filter(data_file_path, dcora_subdir)
            else:
                logger.info(
                    f"DCORA results exist for data file: {data_file}. Skipping DCORA filter."
                )

            if is_dir_empty(evo_subdir):
                logger.info(f"Applying EVO filter to data file: {data_file} ...")
                self._evo_filter(
                    cora_subdir, dcora_subdir, ground_truth_subdir, evo_subdir
                )
            else:
                logger.info(
                    f"EVO results exist for data file: {data_file}. Skipping EVO filter."
                )

        logger.info("Evaluation pipeline complete.")


def main(args):
    parser = argparse.ArgumentParser(
        description="This script is used to compare CORA and DCORA on datasets obeying PyFG formatting."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="directory containing *.pfyg data files for evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="directory where evaluation results are saved",
    )
    parser.add_argument(
        "--override_cora_results",
        action="store_true",
        help="Flag to override existing cora results",
    )
    parser.add_argument(
        "--override_dcora_results",
        action="store_true",
        help="Flag to override existing dcora results",
    )
    parser.add_argument(
        "--override_ground_truth_results",
        action="store_true",
        help="Flag to override existing ground truth results",
    )
    parser.add_argument(
        "--override_evo_results",
        action="store_true",
        help="Flag to override existing evo results",
    )

    args = parser.parse_args()
    evaluation_pipeline = EvaluationPipeline(args)
    evaluation_pipeline.evaluate()


if __name__ == "__main__":
    main(sys.argv[1:])
