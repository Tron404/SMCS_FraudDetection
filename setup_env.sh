device_flag=""
cuda_flag=""

while getopts "d:c:" flag; do
    case "${flag}" in
        d) device_flag="${OPTARG}";;
        c) cuda_flag="${OPTARG}";;
    esac
done

if [[ $device_flag == "cpu" ]]; then
    pip install -r requirements_cpu.txt
elif [[ $device_flag == "gpu" ]]; then
    if [[ $cuda_flag == "" ]]; then
        error_line="Error: provide a cuda version when using the GPU."
        >&2 echo -e "\e[01;31m$error_line\e[0m"
        exit 1
    fi

    # default packages, with no special versioning
    pip install seaborn scikit-learn pandas

    # torch-related packages
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/${cuda_flag}
    pip install torch_geometric
    pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.4.0+${cuda_flag}.html
fi