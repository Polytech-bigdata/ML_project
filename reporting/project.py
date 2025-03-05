#test code from evidently github
import datetime
import pandas as pd

from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, ColumnSummaryMetric,DatasetDriftMetric,DatasetMissingValuesMetric
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.ui.dashboards import CounterAgg,DashboardPanelCounter,DashboardPanelPlot,PanelValue,PlotType,ReportFilter
from evidently.ui.workspace import Workspace,WorkspaceBase
from evidently.metric_preset import ClassificationPreset
from evidently.metrics.custom_metric import CustomValueMetric
from evidently.base_metric import InputData
from evidently.renderers.html_widgets import WidgetSize

from prometheus_client import Gauge, start_http_server
from sklearn.metrics import balanced_accuracy_score
from scripts.utils import load_pipeline

REF_DATA_FILEPATH = 'data/ref_data.csv'
PROD_DATA_FILEPATH = 'data/prod_data.csv'

WORKSPACE = "workspace"
PROJECT_NAME = "ICU Mortality Prediction"
PROJECT_DESCRIPTION = "BigData project using a kaggle dataset for hospital mortality prediction."

#for analizing the drift about the most interesting features in the dataset
NB_INTERESTING_FEATURES = 5

ref_data = pd.read_csv(REF_DATA_FILEPATH, delimiter=";")
model = load_pipeline("artifacts/model.pkl")
ref_data["prediction"] = model.predict(ref_data.iloc[:, :-1])
prod_data = pd.read_csv(PROD_DATA_FILEPATH, delimiter=";")
features = pd.read_csv("data/features_sorted_by_importance.csv")

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

#create a metric function for balanced_accuracy (both current and ref data) to display the value in the dashboard
def current_balanced_accuracy_func(data: InputData): 
    return balanced_accuracy_score(data.current_data[data.column_mapping.target],
        data.current_data[data.column_mapping.prediction])

def ref_balanced_accuracy_func(data: InputData): 
    return balanced_accuracy_score(data.reference_data[data.column_mapping.target],
        data.reference_data[data.column_mapping.prediction])
    

def create_report():
    data_drift_report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ClassificationPreset(),#contains all the classification metrics (Accuracy, F1 score, precision, recall) and display the confusion matrix
            CustomValueMetric(func=current_balanced_accuracy_func, title="Current : Balanced Accuracy", size=WidgetSize.HALF),
            CustomValueMetric(func=ref_balanced_accuracy_func, title="Reference : Balanced Accuracy", size=WidgetSize.HALF),
            *drift_columns,
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
    project = workspace.create_project(PROJECT_NAME)
    project.description = PROJECT_DESCRIPTION
    
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

    features_values = []
    for i in range(0, NB_INTERESTING_FEATURES):
        features_values.append(
            PanelValue(
                metric_id="ColumnDriftMetric",
                metric_args={"column_name.name": features_list[i]},
                field_path=ColumnDriftMetric.fields.drift_score,
                legend=features_list[i],
            )
        )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title=f"Drift for the {NB_INTERESTING_FEATURES} most important dataset features (Wasserstein drift distance)",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=features_values,
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

    # Démarrer le serveur Prometheus sur le port 8000
    server, t = start_http_server(8000)
    server.serve_forever()

if __name__ == "__main__":
    create_demo_project(WORKSPACE)