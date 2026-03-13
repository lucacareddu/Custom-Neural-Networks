from custom_rul_prediction import infer, data_pipeline

import numpy as np
import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from evidently.ui.workspace import CloudWorkspace

import json


ws = CloudWorkspace(token="sk_prod.019caa1a-c7f1-71b5-a4db-6ea62a098a96.YV9E_FV7celRMHznngtvhCE7OiKbdMhBWajAxzKnCxgekJcDojhGYKpPwuh-iHPgH-kTxCNxSu3CM8rp6EsBIYdYMsbj2uQZoNDNG42Vt_7IFZTTs3DmDJpZZ4_WhL1q", url="https://app.evidently.cloud")

X_train, Y_train, X_test, Y_test = data_pipeline()

cols = ['Type', 'Air temperature [K]','Process temperature [K]', 'Rotational speed [rpm]', 
            'Torque [Nm]', 'delta_temperature', 'power', 'delta_wear', 'id']

X_train = pd.DataFrame(X_train.flatten(end_dim=1).numpy(), columns=cols)
X_test = pd.DataFrame(X_test.flatten(end_dim=1).numpy(), columns=cols)

# pred_test = pd.DataFrame(np.array(infer()), columns=["testt"]).reset_index().head(3)
# pred_infer = pd.DataFrame(np.array(infer(json.dumps(json.load(open('stream_test'))[:23]))), columns=["infer"]).reset_index()

# X_train["series_ids"] = np.repeat(np.arange(len(X_train)), 10)
# X_train["timestamps"] = np.tile(np.arange(10), len(X_train))

report = Report([
    DataDriftPreset(),
    DataSummaryPreset()
])

my_eval = report.run(X_train, X_test)
ws.add_run(run=my_eval, name="My Data Drift Report", project_id="019caabc-bd55-7447-8a6d-c6b59a9b9aa3")
