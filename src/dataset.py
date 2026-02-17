import json
import csv
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Any, Tuple, List, Set 


class CrossTaskVideoDataset(Dataset):
    def __init__(
        self,
        data_root=None,
        json_path=None,
        videos_csv_path=None,
        tasks_primary_path=None,
        tasks_related_path=None,
        actions_vocab_path=None,
        remove_o=True
    ):
        """
        Dataset for CrossTask video predictions enriched with task metadata.

        Args:
            data_root (str or Path, optional):
                Root directory containing CrossTask files.
                Expected structure:
                    data_root/
                        videoclip_preds.json
                        videos.csv
                        tasks_primary.txt
                        tasks_related.txt

            json_path (str, optional): Explicit path to videoclip_preds.json.
            videos_csv_path (str, optional): Explicit path to videos.csv.
            tasks_primary_path (str, optional): Explicit path to tasks_primary.txt.
            tasks_related_path (str, optional): Explicit path to tasks_related.txt.
            remove_o (bool): Whether to remove the "O" (background) class.
        """

        if data_root is not None:
            data_root = Path(data_root)

            json_path = json_path or data_root / "videoclip_preds.json"
            videos_csv_path = videos_csv_path or data_root / "videos.csv"
            tasks_primary_path = tasks_primary_path or data_root / "tasks_primary.txt"
            tasks_related_path = tasks_related_path or data_root / "tasks_related.txt"
            actions_vocab_path = actions_vocab_path or data_root / "vocabs" / "actions.txt"

        # Ensure all required paths are defined
        if not all([json_path, videos_csv_path, tasks_primary_path, tasks_related_path, actions_vocab_path ]):
            raise ValueError(
                "You must either provide data_root or explicitly specify all file paths."
            )

        # Convert to Path objects
        json_path = Path(json_path)
        videos_csv_path = Path(videos_csv_path)
        tasks_primary_path = Path(tasks_primary_path)
        tasks_related_path = Path(tasks_related_path)
        actions_vocab_path = Path(actions_vocab_path)

        # Load video_id to task_id mapping
        self.video_to_task = self._load_video_task_mapping(videos_csv_path)

        # Load tasks metadata
        self.tasks = {}
        self.tasks.update(self._load_tasks_metadata(tasks_primary_path))
        self.tasks.update(self._load_tasks_metadata(tasks_related_path))

        self.admissible_actions, self.admissible_actions_set = self._load_actions_vocab(actions_vocab_path)

        # Load frame-level predictions
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.video_preds = {}
        self.video_true = {}

        for item in raw_data:
            vid = item["idx"]

            preds_collapsed = self._collapse_sequence(item["y_pred"], remove_o)
            true_collapsed = self._collapse_sequence(item["y_true"], remove_o)

            self.video_preds[vid] = preds_collapsed
            self.video_true[vid] = true_collapsed

        self.video_ids = list(self.video_preds.keys())

    def get_context(self, key: str) -> Dict[str, Any]:
        """Fetch context meta by ctx_key (task_id by default)."""
        return self.tasks[key]

    def _load_actions_vocab(self, path: Path) -> Tuple[List[str], Set[str]]:
        """
        Load admissible actions from CrossTask vocab file, e.g.:

          @@UNKNOWN@@
          stir mixture<|eoa|>
          <|sact|><|eoa|>
          <|eact|><|eoa|>
          whisk mixture<|eoa|>
          ...

        Rules:
          - strip whitespace
          - remove trailing '<|eoa|>'
          - drop special tokens: '@@UNKNOWN@@', '<|sact|>', '<|eact|>' (and empty)
          - de-duplicate while preserving order
        """
        special = {"@@UNKNOWN@@", "<|sact|>", "<|eact|>", "<|pad|>"}
        actions: List[str] = []
        seen: Set[str] = set()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue

                # remove eoa marker (if present)
                s = s.replace("<|eoa|>", "").strip()
                if not s or s in special:
                    continue

                if s not in seen:
                    actions.append(s)
                    seen.add(s)

        return actions, seen

    def _collapse_sequence(self, seq, remove_o=True):
        """
        Collapse consecutive identical labels into a single label.
        Optionally remove the background class "O".
        """
        collapsed = []
        prev = None

        for label in seq:
            if remove_o and label == "O":
                prev = label
                continue

            if label != prev:
                collapsed.append(label)

            prev = label

        return collapsed

    def _load_video_task_mapping(self, path):
        """
        Load mapping from video_id to task_id from videos.csv.
        Format:
            task_id,video_id,url
        """
        mapping = {}

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                task_id, video_id, _ = row
                mapping[video_id] = task_id

        return mapping

    def _load_tasks_metadata(self, path):
        """
        Load task metadata from tasks_primary.txt or tasks_related.txt.

        Block format:
            task_id
            task_name
            url
            num_steps
            step1,step2,...
        """
        tasks = {}

        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        i = 0
        while i < len(lines):
            task_id = lines[i]
            task_name = lines[i + 1]
            url = lines[i + 2]
            num_steps = int(lines[i + 3])
            steps_line = lines[i + 4]

            steps = [s.strip() for s in steps_line.split(",")]

            tasks[task_id] = {
                "name": task_name,
                "steps": steps,
                "num_steps": num_steps
            }

            i += 5

        return tasks

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        """
        Returns:
            dict containing:
                - video_id
                - task_id
                - task_name
                - task_description
                - pred_sequence
                - true_sequence
        """
        vid = self.video_ids[idx]

        task_id = self.video_to_task.get(vid)
        task_info = self.tasks.get(task_id, {})

        return {
            "video_id": vid,
            "task_id": task_id,
            "task_name": task_info.get("name"),
            "task_description": task_info.get("steps"),
            "pred_sequence": self.video_preds[vid],
            "true_sequence": self.video_true[vid],
        }
