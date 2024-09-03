#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
import logging

import urllib3

logging.captureWarnings(True)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
