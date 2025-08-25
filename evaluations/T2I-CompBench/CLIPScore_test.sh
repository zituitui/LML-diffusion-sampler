#!/bin/bash

save_dir=$1
model_list=($(ls -d "$save_dir"/*/ | xargs -n1 basename))

for model in "${model_list[@]}"
do
    # attr_list=($(ls -d "$save_dir/$model"/*/ | xargs -n1 basename))
    attr_list=("complex" "non_spatial")
    for attr in "${attr_list[@]}"
    do
        sampler_list=($(ls -d "$save_dir/$model/$attr"/*/ | xargs -n1 basename))
        for sampler in "${sampler_list[@]}"
        do
            out_dir="$save_dir/$model/$attr/$sampler"
            # run python script
            echo "Running for model=$model, attr=$attr, sampler=$sampler"
            if [ "$attr" = "complex" ]; then
                python CLIPScore_eval/CLIP_similarity.py --outpath="$out_dir" --complex=True
            else
                python CLIPScore_eval/CLIP_similarity.py --outpath="$out_dir"
            fi
            
            # check if the command was successful
            if [ $? -ne 0 ]; then
                failed_runs+=("model=$model, attr=$attr, sampler=$sampler")
            else
                success_runs+=("model=$model, attr=$attr, sampler=$sampler")
            fi
        done
    done
done

# print count of runs
echo "Total runs: ${#success_runs[@]} succeeded, ${#failed_runs[@]} failed."

# print all failed runs
if [ ${#failed_runs[@]} -ne 0 ]; then
    echo "The following runs failed:"
    for run in "${failed_runs[@]}"
    do
        echo "$run"
    done
else
    echo "All runs completed successfully."
fi