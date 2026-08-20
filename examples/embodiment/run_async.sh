#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_async.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_sac_mlp_async"
else
    CONFIG_NAME=$1
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${ROBOT_PLATFORM:-"LIBERO"}
PROFILE_UTIL=0
EXTRA_CLI_OVERRIDES=""
ROBOT_PLATFORM_SET=0
for ARG in "${@:2}"; do
    case "$ARG" in
        profile-util|--profile-util)
            PROFILE_UTIL=1
            ;;
        *=*|+*)
            EXTRA_CLI_OVERRIDES+=" $ARG"
            ;;
        *)
            if [ "$ROBOT_PLATFORM_SET" -eq 0 ]; then
                ROBOT_PLATFORM=$ARG
                ROBOT_PLATFORM_SET=1
            else
                EXTRA_CLI_OVERRIDES+=" $ARG"
            fi
            ;;
    esac
done

export ROBOT_PLATFORM
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}${EXTRA_CLI_OVERRIDES}"
echo ${CMD} > ${MEGA_LOG_FILE}

generate_overlap_report() {
    local matrix_dir
    local experiment_dir
    local nsys_dir
    local report_status
    local -a report_cmd

    matrix_dir=$(find "${LOG_DIR}" -type f \
        -path '*/overlap_profile/matrix/pair_samples.jsonl' \
        -size +0c -printf '%h\n' -quit 2>/dev/null)
    if [ -z "${matrix_dir}" ]; then
        return
    fi

    experiment_dir=$(dirname "$(dirname "${matrix_dir}")")
    nsys_dir="${experiment_dir}/profiling"
    report_cmd=(
        python -m toolkits.overlap_profiler report
        --run-dir "${matrix_dir}"
    )
    if [ -n "$(find "${nsys_dir}" -type f -name '*.nsys-rep' -print -quit 2>/dev/null)" ]; then
        report_cmd+=(--nsys-dir "${nsys_dir}")
    fi

    echo "Generating overlap profile report from ${matrix_dir}" | tee -a "${MEGA_LOG_FILE}"
    "${report_cmd[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
    report_status=${PIPESTATUS[0]}
    if [ "${report_status}" -ne 0 ]; then
        echo "Warning: overlap report generation failed with status ${report_status}" \
            | tee -a "${MEGA_LOG_FILE}" >&2
    fi
}

if [ "$PROFILE_UTIL" -eq 1 ]; then
    UTIL_OUTPUT_DIR="${LOG_DIR}/hardware_util"
    UTIL_INTERVAL=${UTIL_PROFILE_INTERVAL:-1.0}
    PROFILE_CMD="python ${REPO_PATH}/toolkits/hardware_monitor/record_utilization.py -o ${UTIL_OUTPUT_DIR} -i ${UTIL_INTERVAL} --title ${CONFIG_NAME}_hardware_util -- ${CMD}"
    echo ${PROFILE_CMD} >> ${MEGA_LOG_FILE}
    ${PROFILE_CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
    RUN_STATUS=${PIPESTATUS[0]}
else
    ${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
    RUN_STATUS=${PIPESTATUS[0]}
fi

if [ "${RUN_STATUS}" -eq 0 ]; then
    generate_overlap_report
fi

exit "${RUN_STATUS}"
