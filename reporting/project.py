#test code from evidently github
import datetime
import pandas as pd

from evidently import ColumnMapping
from evidently.future.datasets import DataDefinition, Dataset
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently.metrics import ClassificationConfusionMatrix
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
from evidently.metric_preset import ClassificationPreset

from scripts.utils import load_heterogeneous_dataset, load_pipeline

REF_DATA_FILEPATH = 'data/ref_data.csv'
PROD_DATA_FILEPATH = 'data/prod_data.csv'

WORKSPACE = "workspace"
YOUR_PROJECT_NAME = "ICU Mortality Prediction"
YOUR_PROJECT_DESCRIPTION = "BigData project using a kaggle dataset for hospital mortality prediction."

#for analizing the drift about the most interesting features in the dataset
NB_INTERESTING_FEATURES = 5

ref_data = pd.read_csv(REF_DATA_FILEPATH, delimiter=";")
model = load_pipeline("artifacts/model.pkl")
ref_data["prediction"] = model.predict(ref_data.iloc[:, :-1])
prod_data = pd.read_csv(PROD_DATA_FILEPATH, delimiter=";")
# ref_data , ref_y, ref_col_num, ref_col_cat, ref_labels = load_heterogeneous_dataset(REF_DATA_FILEPATH, delimiter=";", predict_var=False)
# prod_data, prod_y, prod_col_num, prod_col_cat, prod_labels = load_heterogeneous_dataset(PROD_DATA_FILEPATH, delimiter=";", predict_var=False)
# prod_data = prod_data.iloc[:, :-1]
features = pd.read_csv("data/features_sorted_by_importance.csv")


#define metrics used in the report
clf_metrics = {
    'F1 score' : "f1",
    'Balanced Accuracy' : "balanced_accuracy",
    'Precision' : "precision",
    'Recall' : "recall",
}

#define columns on which drift are interesting to monitor
features_list = features.iloc[:, 0].tolist()
drift_columns = []
for i in range(0, NB_INTERESTING_FEATURES):
    drift_columns.append(ColumnDriftMetric(column_name=features_list[i], stattest="wasserstein"))
    drift_columns.append(ColumnSummaryMetric(column_name=features_list[i]))

column_mapping = ColumnMapping(
    target='target',   
    prediction='prediction', 
)

def create_report():
    data_drift_report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ClassificationConfusionMatrix(),
            *drift_columns,
            ClassificationPreset(),
        ],
        timestamp=datetime.datetime.now(),
    )

    data_drift_report.run(reference_data=ref_data, current_data=prod_data, column_mapping=column_mapping)
    return data_drift_report


def create_test_suite():
    data_drift_test_suite = TestSuite(
        tests=[DataDriftTestPreset()],
        timestamp=datetime.datetime.now(),
    )

    data_drift_test_suite.run(reference_data=ref_data, current_data=prod_data)
    return data_drift_test_suite


def create_project(workspace: WorkspaceBase):
    project = workspace.create_project(YOUR_PROJECT_NAME)
    project.description = YOUR_PROJECT_DESCRIPTION

    #add metrics panels to the dashboard
    for metric_name in clf_metrics.keys():
        project.dashboard.add_panel(
            DashboardPanelCounter(
                title=metric_name,
                filter=ReportFilter(metadata_values={}, tag_values=[]),
                value=PanelValue(
                    metric_id="ClassificationQualityMetric",
                    metric_args= {"metric": clf_metrics[metric_name]},
                    field_path="value",               
                    legend=metric_name,
                ),
                text=metric_name,
                agg=CounterAgg.LAST,
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
                PanelValue(
                    metric_id="ClassificationQualityMetric",
                    metric_args={"metric": "f1"},
                    field_path="value",
                    legend="F1 Score",
                ),
                PanelValue(
                    metric_id="ClassificationQualityMetric",
                    metric_args={"metric": "balanced_accuracy"},
                    field_path="value",
                    legend="Balanced Accuracy",
                ),
            ],
            plot_type=PlotType.LINE,
        )
    )

    for i in range(0, NB_INTERESTING_FEATURES):
        title = str(features_list[i]) + " : Wasserstein drift distance"
        column_name = features_list[i]
        project.dashboard.add_panel(
            DashboardPanelPlot(
                title=title,
                filter=ReportFilter(metadata_values={}, tag_values=[]),
                values=[
                    PanelValue(
                        metric_id="ColumnDriftMetric",
                        metric_args={"column_name.name": column_name},
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

    report = create_report()
    ws.add_report(project.id, report)

    test_suite = create_test_suite()
    ws.add_test_suite(project.id, test_suite)


if __name__ == "__main__":
    create_demo_project(WORKSPACE)