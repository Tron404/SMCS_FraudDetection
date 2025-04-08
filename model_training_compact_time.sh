# neighbourhood = how many neighbouring nodes to sample in the ith hop (e.g. [10, 5] --> 10 nodes in the first hop and 5 in the second hop)
# model = which model architecture to use (choices: sage_attn, gcn, gat, sage)
# dataset = which dataset to train on (choices: elliptic++, dgraphfin)
# device = on which device to train/load data (choices: cpu, cuda)
# epochs = for how many epochs to train a model

PROCESS_LIMIT=3

find_max_idx() {
    local max=-1
    for num_idx in $@; do
        if (( $num_idx > $max )); then
            max=$num_idx
        fi
    done

    echo $max
}

experiment_loop() {
    local process_counter=0
    local model=$1
    local -n _neighbourhood_sizes=$2
    local path=$3
    local lr=1e-5

    if [[ $model == "sage_attn" ]];then
        lr=5e-5
    fi

    for n_size in ${_neighbourhood_sizes[@]}; do
        for train_run_idx in 0 1 2; do
            new_path="$path$n_size/$train_run_idx"
            mkdir -p $new_path
            echo $new_path
            python training_script_time.py --neighbourhood $n_size --epochs 10 --model $model --dataset "elliptic++" --device cuda:0 --path $new_path --lr $lr &
            process_counter=$(($process_counter+1))
            if (($process_counter >= $PROCESS_LIMIT)); then
                process_counter=0
                wait
            fi
        done
        wait
    done
}

models=("sage")
for model in ${models[@]}; do
    path=""./experiments/"model_checkpoints_final_time/${model^^}"
    mkdir -p $path

    PROCESS_LIMIT=3
    neighbourhood_sizes=("(25,10)" "(50,10)" "(100,10)" "(250,10)" "(500,10)" "(1000,10)")
    experiment_loop $model neighbourhood_sizes "$path/"

    PROCESS_LIMIT=2
    neighbourhood_sizes=("(-1)" "(-1,-1)")
    experiment_loop $model neighbourhood_sizes "$path/"
done