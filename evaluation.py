from evalutils import ClassificationEvaluation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import json
import pandas as pd
import glob
import os
import logging
from typing import List

LOG = logging.getLogger(__name__)
logging.basicConfig(format='[%(filename)s:%(lineno)s - %(funcName)20s()] %(asctime)s %(message)s', level="INFO")


class SurgVU(ClassificationEvaluation):
    def __init__(self):
        self._gt_json_list = self.get_gt_json_list()
        self._pred_json_list = self.get_pred_json_list()
        logging.info(f"Found the following ground truth JSON files: {self._gt_json_list}")
        logging.info(f"Found the following prediction JSON files: {self._pred_json_list}")
        self._label_encoder = self.init_label_encoder()

    def init_label_encoder(self) -> LabelEncoder:
        labels = [
            "none",
            "range_of_motion",
            "rectal_artery_vein",
            "retraction_collision_avoidance",
            "skills_application",
            "suspensory_ligaments",
            "suturing",
            "uterine_horn"
            "other"
        ]
        return LabelEncoder().fit(labels)

    def get_gt_json_list(self) -> list:
        return glob.glob(os.path.join(os.getcwd(), "true_jsons", "*.json"))
    
    def get_pred_json_list(self) -> list:
        return glob.glob(os.path.join(os.getcwd(), "pred_jsons", "*.json"))

    def evaluate_all_gt(self, gt_json_list: List[str]) -> pd.DataFrame:
        results = []
        for gt_json in gt_json_list:
            gt_data = self.load_json(gt_json)
            pred_json = self.find_corresponding_pred(gt_json)
            if pred_json is None:
                LOG.warning(f"No corresponding prediction JSON found for ground truth JSON: {gt_json}")
                continue
            pred_data = self.load_json(pred_json)
            result = self.evaluate_single_video(gt_data, pred_data)
            results.append(result)
        return pd.DataFrame(results)
    
    def find_corresponding_pred(self, gt_json: str) -> str:
        basename = os.path.basename(gt_json)
        pred = list(filter(lambda x: basename in os.path.basename(x), self._pred_json_list))
        if len(pred) == 0:
            return None
        if len(pred) > 1:
            LOG.warning(f"Multiple corresponding prediction JSON found for ground truth JSON: {gt_json}")
        return pred[0]
    
    def load_json(self, json_file: str) -> dict:
        with open(json_file, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def evaluate_single_video(self, gt_data: pd.DataFrame, pred_data: pd.DataFrame) -> dict:
        combined = pd.merge(gt_data, pred_data, on="slice_nr", how="left")
        combined['step_label_y'].fillna("none")
        combined['true'] = self._label_encoder.transform(combined['step_label_x'])
        combined['pred'] = self._label_encoder.transform(combined['step_label_y'])
        results = {
            "accuracy": accuracy_score(combined['true'], combined['pred']),
            "f1": f1_score(combined['true'], combined['pred'], average="weighted"),
            "precision": precision_score(combined['true'], combined['pred'], average="weighted"),
            "recall": recall_score(combined['true'], combined['pred'], average="weighted")
        }
        return results

    def evaluate(self):
        results = self.evaluate_all_gt(self._gt_json_list)
        summary_dict = {
            "accuracy": float(results['accuracy'].mean()),
            "f1": float(results['f1'].mean()),
            "precision": float(results['precision'].mean()),
            "recall": float(results['recall'].mean())
        }
        if os.path.isdir("/output/"):
            filename = "/output/metrics.json"
            LOG.info("Writing evaluation results to /output/metrics.json")
            with open(filename, 'w') as f:
                json.dump(summary_dict, f, indent=4)
        else:
            LOG.info("No /output/ directory found, evaluation results will not be written to a file.")
            LOG.info(summary_dict)



if __name__ == "__main__":
    evaluator = SurgVU()
    evaluator.evaluate()