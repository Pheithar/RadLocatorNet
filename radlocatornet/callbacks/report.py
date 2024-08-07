from lightning import Callback, LightningModule, Trainer
from radlocatornet.models import RadLocatorNetworkModule
from radlocatornet.reports import Report
import os
import numpy as np


class ReportCallback(Callback):
    """
    Callback that compiles all the losses during training, and at the end creates a custom report with the losses, plots and any other information that is needed.

    The report is generated at the end of the test step, which is the last step of the training.

    Attributes:
        save_path (str): The path to save the report
        name (str): The name of the report
    """

    save_path: str
    name: str

    def __init__(self, save_path: str, name: str) -> None:
        """
        Initialize the callback. The save path is the path to save the report.

        Args:
            save_path (str): The path to save the report
            name (str): The name of the report
        """
        self.save_path = save_path
        self.name = name

        self.train_metrics = {}
        self.validation_metrics = {}
        self.test_metrics = {}

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Extract the metrics at the end of the epoch

        Args:
            trainer (Trainer): The trainer that is running the training
            pl_module (LightningModule): The LightningModule that is being trained
        """
        self._extract_metrics(trainer)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """Extract the metrics at the end of the validation step

        Args:
            trainer (Trainer): The trainer that is running the training
            pl_module (LightningModule): The LightningModule that is being trained
        """
        self._extract_metrics(trainer)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Extract the metrics at the end of the test step

        Args:
            trainer (Trainer): The trainer that is running the training
            pl_module (LightningModule): The LightningModule that is being trained
        """
        self._extract_metrics(trainer)

    def on_test_end(self, trainer: Trainer, pl_module: RadLocatorNetworkModule) -> None:
        """Create and save the report at the end of the test step

        Args:
            trainer (Trainer): The trainer that is running the training
            pl_module (RadLocatorNetworkModule): The RadLocatorNetworkModule that is being trained
        """
        report = Report(self.name)

        title_font = ("helvetica", "B", 16)
        normal_font = ("helvetica", "", 10)
        table_header_font = ("helvetica", "B", 10)
        small_font = ("helvetica", "", 8)

        page_width = report.w - 2 * report.l_margin - 10

        # Title
        report.set_font(*title_font)
        report.cell(0, 10, "Model Summary", ln=True)

        # Small fontsize
        report.set_font(*small_font)
        report.multi_cell(0, 3, pl_module.__repr__(), ln=True)

        report.set_font(*title_font)
        report.cell(0, 10, "Dataset Summary", ln=True)

        report.set_font(*small_font)
        report.multi_cell(0, 3, trainer.datamodule.dataset.__repr__(), ln=True)

        # Print a table with all the metrics, with columns for Train, Validation, and Test, and rows for each metric
        report.set_font(*title_font)
        report.cell(0, 10, "Metrics", ln=True)

        report.set_font(*table_header_font)
        report.set_fill_color(200, 200, 200)
        report.cell(page_width / 4, 8, "Metric", border=True, ln=False, fill=True)
        report.cell(
            page_width / 4, 8, "Train", border=True, ln=False, fill=True, align="C"
        )
        report.cell(
            page_width / 4,
            8,
            "Validation",
            border=True,
            ln=False,
            fill=True,
            align="C",
        )
        report.cell(
            page_width / 4, 8, "Test", border=True, ln=True, fill=True, align="C"
        )

        report.set_font(*normal_font)

        # NOTE: If the metric is not present in the train dictionary, then it will not be shown in the report. If the metric is not present in the validation or test dictionary, then it will be shown as "-".

        for metric_name in self.train_metrics.keys():
            report.cell(page_width / 4, 8, metric_name, border=True, ln=False)
            report.cell(
                page_width / 4,
                8,
                self.train_metrics.get(metric_name, "-"),
                border=True,
                ln=False,
                align="C",
            )
            report.cell(
                page_width / 4,
                8,
                self.validation_metrics.get(metric_name, "-"),
                border=True,
                ln=False,
                align="C",
            )
            report.cell(
                page_width / 4,
                8,
                self.test_metrics.get(metric_name, "-"),
                border=True,
                ln=True,
                align="C",
            )

        # If the folder does not exist, then create the folder.
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        # If the file is a directory, then the report will be saved in the directory.
        if os.path.isdir(self.save_path):
            self.save_path = os.path.join(self.save_path, "report.pdf")

        report.output(self.save_path)

    def _extract_metrics(self, trainer: Trainer) -> None:
        """Extract the metrics from the model. All metrics should be store in the model using the `self.log` method. This method will extract the metrics from the model and store them in the callback.

        The metrics are always ``Train [metric name]``, ``Validation [metric name]``, and ``Test [metric name]``.

        Args:
            trainer (Trainer): The trainer that is running the training
        """
        for metric_name, metric_value in trainer.logged_metrics.items():
            if "Train" in metric_name:
                metric_name = metric_name.replace("Train", "").strip()
                self.train_metrics[metric_name] = np.format_float_scientific(
                    metric_value.item(), precision=2
                )

            elif "Validation" in metric_name:
                metric_name = metric_name.replace("Validation", "").strip()
                self.validation_metrics[metric_name] = np.format_float_scientific(
                    metric_value.item(), precision=2
                )

            elif "Test" in metric_name:
                metric_name = metric_name.replace("Test", "").strip()
                self.test_metrics[metric_name] = np.format_float_scientific(
                    metric_value.item(), precision=2
                )
