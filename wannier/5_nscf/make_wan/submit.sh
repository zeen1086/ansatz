for i in {1..44..4}
do
    j=$((i + 3))
    if [ "$j" -gt 44 ]; then
        j=44
    fi

    echo "$i $j"
    cp sub_run.job sub_run_tmp.job
    sed -i "s/HYC1/${i}/g" sub_run_tmp.job
    sed -i "s/HYC2/${j}/g" sub_run_tmp.job
    qsub sub_run_tmp.job
    sleep 1
done