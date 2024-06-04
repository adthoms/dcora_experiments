import sys
import argparse
import attr
import os
import numpy as np
import logging
from py_factor_graph.io.pyfg_file import read_from_pyfg_file
from py_factor_graph.io.tum_file import save_robot_trajectories_to_tum_file
from py_factor_graph.utils.logging_utils import logger
logging.basicConfig(level=logging.INFO)


class EvaluationPipeline:
    def __init__(self, args):
      self.data_dir = args.data_dir
      self.output_dir = args.output_dir

    def evaluate(self) -> None:
      logger.info(f"Evaluation Pipeline starting...")
      self.process_data_files()
      logger.info(f"Evaluation Pipeline finished.")

    def process_data_files(self) -> None:
      data_file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.pyfg')]
      for data_file in data_file_list:
          # create separate directory for each data file
          data_file_out_dir = os.path.join(self.output_dir, data_file)
          if not os.path.exists(data_file_out_dir):
              os.makedirs(data_file_out_dir)

          # save ground truth trajectories to TUM file
          data_file_path = os.path.join(self.data_dir, data_file)
          fg = read_from_pyfg_file(data_file_path)
          save_robot_trajectories_to_tum_file(fg, data_file_out_dir)

          # process data file with CORA
          # TODO(AT): create custom CORA executable that will operate on data_file and save its output to a CORA directory in data_file_out_dir

          # process data file with DCORA
          # TODO(AT): create custom DCORA executable that will operate on data_file and save its output to a DCORA directory in data_file_out_dir

          # compare CORA and DCORA against ground truth
          # TODO(JV): Use EVO for comparison. Consider using subprocess to call EVO from the command line.


def main(args):
    parser = argparse.ArgumentParser(
        description='This script is used to compare CORA and DCORA on benchmark datasets obeying PyFG formatting.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory containing *.pfyg data files for comparison')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='directory containing TUM formatted trajectories for comparison')

    # parse args and process
    args = parser.parse_args()
    evaluation_pipeline = EvaluationPipeline(args)
    evaluation_pipeline.evaluate()


if __name__ == "__main__":
    main(sys.argv[1:])