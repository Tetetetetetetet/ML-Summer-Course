import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("read_data.log",mode="w")
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
log = logger


class AdmissionType(Enum):
    Emergency = 1
    Urgent = 2
    Elective = 3
    Newborn = 4
    NotAvailable = 5
    NULL = 6
    TraumaCenter = 7
    NotMapped = 8

class DischargeDisposition(Enum):
    DischargedToHome = 1
    DischargedToShortTermHospital = 2
    DischargedToSNF = 3
    DischargedToICF = 4
    Expired = 11
    HospiceHome = 13
    HospiceMedicalFacility = 14
    NotMapped = 25
    UnknownInvalid = 26
    # 可按需补全其他项

class AdmissionSource(Enum):
    PhysicianReferral = 1
    ClinicReferral = 2
    HMOReferral = 3
    TransferFromHospital = 4
    EmergencyRoom = 7
    NotAvailable = 9
    TransferFromCriticalAccessHospital = 10
    NULL = 17
    NotMapped = 20
    UnknownInvalid = 21
    # 继续添加其他值...

def read_data():
    # 全部显示设置（不会被截断）
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", 1000)
    dataset_path = Path(__file__).parent.parent / "Dataset"
    test_data=pd.read_csv(dataset_path /"diabetic_data_test.csv")
    train_data=pd.read_csv(dataset_path / "diabetic_data_training.csv")
    log.info("train_data head:\n{}".format(train_data.head()))
    log.info("train data describe:\n{}".format(train_data.describe()))
    log.info("test_data head:\n{}".format(test_data.head()))
    log.info("test data describe:\n{}".format(test_data.describe()))
    return train_data, test_data

train_data, test_data = read_data()
log.info("train_data shape: {}".format(train_data.shape))
log.info("test_data shape: {}".format(test_data.shape))
