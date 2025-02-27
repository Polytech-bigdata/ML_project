#test code from evidently github
import datetime
import pandas as pd

from evidently.future.datasets import DataDefinition, Dataset
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently.metrics import ClassificationQualityMetric
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.ui.dashboards import CounterAgg
from evidently.ui.dashboards import DashboardPanelCounter
from evidently.ui.dashboards import DashboardPanelPlot
from evidently.ui.dashboards import PanelValue
from evidently.ui.dashboards import PlotType
from evidently.ui.dashboards import ReportFilter
from evidently.ui.remote import RemoteWorkspace
from evidently.ui.workspace import Workspace
from evidently.ui.workspace import WorkspaceBase

from scripts.utils import load_heterogeneous_dataset

REF_DATA_FILEPATH = '../data/ref_data.csv'
PROD_DATA_FILEPATH = '../data/prod_data.csv'

WORKSPACE = "workspace"
YOUR_PROJECT_NAME = "ICU Mortality Prediction"
YOUR_PROJECT_DESCRIPTION = "BigData project using a kaggle dataset for hospital mortality prediction."

#for analizing the drift about the most interesting features in the dataset
NB_INTERESTING_FEATURES = 5

df_ref_data , ref_y, ref_col_num, ref_col_cat, ref_labels = load_heterogeneous_dataset(REF_DATA_FILEPATH)
df_prod_data, prod_y, prod_col_num, prod_col_cat, prod_labels = load_heterogeneous_dataset(PROD_DATA_FILEPATH)
features = pd.read_csv("../scripts/features_sorted_by_importance.csv")

schema = DataDefinition(
    numerical_columns=ref_col_num,
    categorical_columns=ref_col_cat
    )

prod_data = Dataset.from_pandas(
    df_prod_data,
    data_definition=schema
)

ref_data = Dataset.from_pandas(
    df_ref_data,
    data_definition=schema
)

#define metrics used in the report
clf_metrics = [
    ClassificationQualityMetric(metric="f1"),
    ClassificationQualityMetric(metric="balanced_accuracy"),
    ClassificationQualityMetric(metric="precision"),
    ClassificationQualityMetric(metric="recall"),
]

#define columns on which drift are interesting to monitor
#drift_columns = ["age", "bmi","ethnicity","gender","height","weight","arf_apache","apache_4a_hospital_death_prob","apache_4a_icu_death_prob","aids","cirrhosis",]
features_list = features.iloc[:, 0].tolist()
drift_columns = []
for i in range(0, NB_INTERESTING_FEATURES):
    drift_columns.append(ColumnDriftMetric(column_name=features_list[i], stattest="wasserstein"))
    drift_columns.append(ColumnSummaryMetric(column_name=features_list[i]))


def create_report(i: int):
    data_drift_report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            *drift_columns,
            *clf_metrics,
            #DataSummaryPreset()
        ],
        timestamp=datetime.datetime.now() + datetime.timedelta(days=i),
    )

    data_drift_report.run(reference_data=ref_data, current_data=prod_data.iloc[100 * i : 100 * (i + 1), :])
    return data_drift_report


def create_test_suite(i: int):
    data_drift_test_suite = TestSuite(
        tests=[DataDriftTestPreset()],
        timestamp=datetime.datetime.now() + datetime.timedelta(days=i),
    )

    data_drift_test_suite.run(reference_data=ref_data, current_data=prod_data.iloc[100 * i : 100 * (i + 1), :])
    return data_drift_test_suite


def create_project(workspace: WorkspaceBase):
    project = workspace.create_project(YOUR_PROJECT_NAME)
    project.description = YOUR_PROJECT_DESCRIPTION

    #add metrics panels to the dashboard
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="F1 score",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                metric_args= {"metric": "f1"},
                field_path="value",               
                legend="F1 score",
            ),
            text="F1 score",
            agg=CounterAgg.LAST,
            size=1,
        )
    )

    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Balanced Accuracy",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                metric_args= {"metric": "balanced_accuracy"},
                field_path="value",               
                legend="Balanced Accuracy",
            ),
            text="Balanced Accuracy",
            agg=CounterAgg.LAST,
            size=1,
        )
    )

    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Recall",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                metric_args= {"metric": "recall"},
                field_path="value",               
                legend="Recall",
            ),
            text="Recall",
            agg=CounterAgg.LAST,
            size=1,
        )
    )

    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Precision",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="ClassificationQualityMetric",
                metric_args= {"metric": "precision"},
                field_path="value",               
                legend="Precision",
            ),
            text="Precision",
            agg=CounterAgg.LAST,
            size=1,
        )
    )

    project.dashboard.add_panel(
        DashboardPanelCounter(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            agg=CounterAgg.NONE,
            title="Census Income Dataset (Adult)",
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Model Calls",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetMissingValuesMetric",
                field_path=DatasetMissingValuesMetric.fields.current.number_of_rows,
                legend="count",
            ),
            text="count",
            agg=CounterAgg.SUM,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Share of Drifted Features",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetDriftMetric",
                field_path="share_of_drifted_columns",
                legend="share",
            ),
            text="share",
            agg=CounterAgg.LAST,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Dataset Quality",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(metric_id="DatasetDriftMetric", field_path="share_of_drifted_columns", legend="Drift Share"),
                PanelValue(
                    metric_id="DatasetMissingValuesMetric",
                    field_path=DatasetMissingValuesMetric.fields.current.share_of_missing_values,
                    legend="Missing Values Share",
                ),
            ],
            plot_type=PlotType.LINE,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Age: Wasserstein drift distance",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "age"},
                    field_path=ColumnDriftMetric.fields.drift_score,
                    legend="Drift Score",
                ),
            ],
            plot_type=PlotType.BAR,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Education-num: Wasserstein drift distance",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "education-num"},
                    field_path=ColumnDriftMetric.fields.drift_score,
                    legend="Drift Score",
                ),
            ],
            plot_type=PlotType.BAR,
            size=1,
        )
    )
    project.save()
    return project


def create_demo_project(workspace: str):
    ws = Workspace.create(workspace)
    project = create_project(ws)

    for i in range(0, 5):
        report = create_report(i=i)
        ws.add_report(project.id, report)

        test_suite = create_test_suite(i=i)
        ws.add_test_suite(project.id, test_suite)


if __name__ == "__main__":
    create_demo_project(WORKSPACE)