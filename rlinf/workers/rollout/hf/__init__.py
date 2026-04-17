# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Import VLM-AE pipeline worker to apply patches if enabled
import os
if os.environ.get("VLM_AE_DISAGG", "0") == "1" or os.environ.get(
    "VLM_AE_BASELINE_MICROPIPE", "0"
) == "1":
    from rlinf.workers.rollout.hf.vlm_ae_pipeline_worker import patch_huggingface_worker_for_pipeline
    patch_huggingface_worker_for_pipeline()
